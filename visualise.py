import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from model import BallHeatmapCNN, extract_centres


def draw_cross(frame, x, y, size=40, color=(0, 255, 0), thickness=4):
    """Draw a large X at (x, y) in original 1920x1080 coords."""
    cv2.line(frame, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(frame, (x + size, y - size), (x - size, y + size), color, thickness)


def predict_video(
    model,
    video_path: str,
    output_path: str,
    device="cuda",
    threshold=0.5,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    print(f"Opened: {cap.isOpened()}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    if not cap.isOpened():
        print(f"ERROR: Could not open video at {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )

    model.eval()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor)[0].squeeze().cpu().numpy()  # (135, 240)

        # Extract centres in heatmap coords, scale back to original
        centres = extract_centres(pred, threshold)
        for (cx, cy) in centres:
            ox = int(cx * 8)
            oy = int(cy * 8)
            draw_cross(frame, ox, oy)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total} frames")

    cap.release()
    out.release()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BallHeatmapCNN(in_channels=3).to(device)
    model.load_state_dict(torch.load("ball_heatmap_cnn.pth", map_location=device))

    predict_video(
        model,
        video_path="clip 4.mp4",
        output_path="clip4_predicted.avi",
        device=device,
        threshold=0.5,
    )