import torch
import time
from model import BallHeatmapCNN

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BallHeatmapCNN(in_channels=3).to(device)
    model.load_state_dict(torch.load("ball_heatmap_cnn.pth", map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, 1080, 1920).to(device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Inference: {(end-start)/100*1000:.1f}ms per frame")
    print(f"Theoretical max: {100/(end-start):.1f}fps")