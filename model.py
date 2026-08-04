from json_to_heatmap import dfs
from json_to_heatmap import dataframe_to_heatmaps
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
from pathlib import Path
from scipy.ndimage import label, center_of_mass
from sklearn.metrics import r2_score


# ── Architecture ───────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BallHeatmapCNN(nn.Module):
    """
    Input:  (B, 3, 1080, 1920)
    Output: (B, 1,  135,  240)  — exactly 8x reduction via 3x MaxPool2d(2)
    """
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, 16, kernel=7, padding=3),
            ConvBNReLU(16, 16),
            nn.MaxPool2d(2),          # 1080x1920 -> 540x960
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(16, 32),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(2),          # 540x960 -> 270x480
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2),          # 270x480 -> 135x240
        )
        self.stage4 = nn.Sequential(
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
        )

        self.heatmap_head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.heatmap_head(x)
        return torch.sigmoid(x)


# ── Dataset ────────────────────────────────────────────────────────────────────

class BallDataset(Dataset):
    """
    Args:
        frame_dir:    path to folder containing frame_xxxxxx.png files
        heatmap_dict: dict mapping frame_index (int) -> np.ndarray (135, 240), float32, [0,1]
    """
    def __init__(self, frame_dir: str, heatmap_dict: dict):
        self.frame_dir = Path(frame_dir)

        available = {
            p.stem
            for p in self.frame_dir.glob("frame_*.png")
        }
        # Strip .png from heatmap keys: "frame_000001.png" -> "frame_000001"
        heatmap_stems = {Path(k).stem: k for k in heatmap_dict.keys()}

        matched = available & set(heatmap_stems.keys())
        self.frame_indices = sorted(matched, key=lambda s: int(s.split("_")[1]))
        self.heatmap_dict = {k: heatmap_dict[heatmap_stems[k]] for k in matched}

        print(f"{self.frame_dir.name}: {len(self.frame_indices)} labelled frames found")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        stem = self.frame_indices[idx]   # "frame_000001"

        path = self.frame_dir / f"{stem}.png"
        frame = cv2.imread(str(path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = self.transform(frame)

        heatmap = self.heatmap_dict[stem]
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).float()

        return frame_tensor, heatmap_tensor


# ── Loss ───────────────────────────────────────────────────────────────────────

def weighted_mse_loss(pred, target, pos_weight=10.0):
    weight = 1.0 + (pos_weight - 1.0) * target
    return (weight * (pred - target) ** 2).mean()

# ── Training ───────────────────────────────────────────────────────────────────

def train(
    frame_dirs: list[str],
    heatmap_dicts: list[dict],
    num_epochs: int = 20,
    batch_size: int = 2,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    datasets = [BallDataset(fd, hd) for fd, hd in zip(frame_dirs, heatmap_dicts)]
    full_dataset = ConcatDataset(datasets)
    print(f"Total labelled frames across all clips: {len(full_dataset)}")

    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = BallHeatmapCNN(in_channels=3).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for frames, heatmaps in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            frames   = frames.to(device)
            heatmaps = heatmaps.to(device)

            optimiser.zero_grad()

            with torch.amp.autocast('cuda'):
                preds = model(frames)
                loss  = weighted_mse_loss(preds, heatmaps)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} — loss: {epoch_loss/len(loader):.6f}")

    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_frame(model, frame_bgr: np.ndarray, device="cuda", threshold=0.5):
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_tensor = transform(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(frame_tensor)

    heatmap = pred[0].squeeze().cpu().numpy()   # (135, 240)
    centres = extract_centres(heatmap, threshold)
    original_coords = [(cx * 8, cy * 8) for cx, cy in centres]

    return heatmap, original_coords

# ── Evaluation ──────────────────────────────────────────────────────────────────

def extract_centres(heatmap_np: np.ndarray, threshold=0.5):
    binary = heatmap_np > threshold
    labeled, n = label(binary)
    centres = []
    for i in range(1, n + 1):
        cy, cx = center_of_mass(heatmap_np, labeled, i)
        centres.append((float(cx), float(cy)))
    return centres

def match_centres(pred_centres, true_centres, max_dist=10.0):
    from scipy.optimize import linear_sum_assignment
    if not pred_centres or not true_centres:
        return [], len(pred_centres), len(true_centres)
    cost = np.array([
        [np.sqrt((px - tx)**2 + (py - ty)**2)
         for tx, ty in true_centres]
        for px, py in pred_centres
    ])
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_dist:
            matched.append((pred_centres[r], true_centres[c]))
    unmatched_pred = len(pred_centres) - len(matched)
    unmatched_true = len(true_centres) - len(matched)
    return matched, unmatched_pred, unmatched_true

def evaluate(
    model,
    frame_dir: str,
    heatmap_dict: dict,
    device="cuda",
    threshold=0.3,
    max_dist=10.0,
):
    model.eval()
    frame_dir = Path(frame_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    heatmap_stems = {Path(k).stem: k for k in heatmap_dict.keys()}
    available = {p.stem for p in frame_dir.glob("frame_*.png")}
    stems = sorted(available & set(heatmap_stems.keys()), key=lambda s: int(s.split("_")[1]))

    print(f"Evaluating on {len(stems)} labelled test frames...")

    all_pred_xy = []
    all_true_xy = []
    distances = []
    false_positives = 0
    false_negatives = 0
    total_true = 0
    total_pred = 0

    for stem in stems:
        frame = cv2.imread(str(frame_dir / f"{stem}.png"))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_heatmap = model(frame_tensor)[0].squeeze().cpu().numpy()

        true_heatmap = heatmap_dict[heatmap_stems[stem]]

        pred_centres = extract_centres(pred_heatmap, threshold)
        true_centres = extract_centres(true_heatmap, threshold)

        matched, unmatched_pred, unmatched_true = match_centres(pred_centres, true_centres, max_dist)

        for (px, py), (tx, ty) in matched:
            all_pred_xy.append((px, py))
            all_true_xy.append((tx, ty))
            distances.append(np.sqrt((px - tx)**2 + (py - ty)**2))

        false_positives += unmatched_pred
        false_negatives += unmatched_true
        total_true += len(true_centres)
        total_pred += len(pred_centres)

    all_pred_xy = np.array(all_pred_xy)
    all_true_xy = np.array(all_true_xy)
    distances   = np.array(distances)

    r2_x = r2_score(all_true_xy[:, 0], all_pred_xy[:, 0])
    r2_y = r2_score(all_true_xy[:, 1], all_pred_xy[:, 1])
    r2   = r2_score(all_true_xy.flatten(), all_pred_xy.flatten())

    mde_heatmap  = distances.mean()
    mde_original = mde_heatmap * 8

    pck_5  = (distances <= 5).mean()
    pck_10 = (distances <= 10).mean()

    precision = len(distances) / max(total_pred, 1)
    recall    = len(distances) / max(total_true, 1)

    print(f"\n── Evaluation Results ─────────────────────────────")
    print(f"Frames evaluated:          {len(stems)}")
    print(f"Total true ball instances: {total_true}")
    print(f"Total pred ball instances: {total_pred}")
    print(f"Matched pairs:             {len(distances)}")
    print(f"")
    print(f"R² (overall):              {r2:.4f}")
    print(f"R² (x):                    {r2_x:.4f}")
    print(f"R² (y):                    {r2_y:.4f}")
    print(f"")
    print(f"Mean Distance Error:       {mde_heatmap:.2f}px (heatmap) / {mde_original:.1f}px (original)")
    print(f"PCK@5px:                   {pck_5*100:.1f}%")
    print(f"PCK@10px:                  {pck_10*100:.1f}%")
    print(f"")
    print(f"Precision:                 {precision*100:.1f}%")
    print(f"Recall:                    {recall*100:.1f}%")
    print(f"False positives:           {false_positives}")
    print(f"False negatives:           {false_negatives}")

    return {
        'r2': r2, 'r2_x': r2_x, 'r2_y': r2_y,
        'mde_heatmap': mde_heatmap, 'mde_original': mde_original,
        'pck_5': pck_5, 'pck_10': pck_10,
        'precision': precision, 'recall': recall,
    }

# ── Entry point ────────────────────────────────────────────────────────────────

heatmap_dicts = []
for raw_df in dfs:
    heatmaps = dataframe_to_heatmaps(raw_df)
    heatmap_dicts.append(heatmaps)

if __name__ == "__main__":
    frame_dirs = [
        "frames1/clip1",
        "frames2/clip2",
        "frames3/clip3",
    ]

    # Your three heatmap dicts: {frame_index (int): np.ndarray (135, 240)}

    model = train(frame_dirs, heatmap_dicts, num_epochs=10, batch_size=4)
    torch.save(model.state_dict(), "ball_heatmap_cnn.pth")
    print("Model saved.")