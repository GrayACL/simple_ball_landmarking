from json_to_heatmap import dfs, dataframe_to_heatmaps
from model import BallHeatmapCNN, evaluate
import torch

if __name__ == "__main__":
    heatmap_dict_test = dataframe_to_heatmaps(dfs[3])   # test is the 4th df

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BallHeatmapCNN(in_channels=3).to(device)
    model.load_state_dict(torch.load("ball_heatmap_cnn.pth", map_location=device))

    metrics = evaluate(
        model,
        frame_dir="test_frames/test",
        heatmap_dict=heatmap_dict_test,
        device=device,
    )