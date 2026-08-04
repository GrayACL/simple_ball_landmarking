import json
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ----------------------------
# Load COCO annotations
# ----------------------------
def load_coco_to_df(path):
    with open(path, "r") as f:
        coco = json.load(f)

    image_map = {img["id"]: img["file_name"] for img in coco["images"]}
    rows = []

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        bbox = ann.get("bbox", None)

        if bbox is None:
            continue

        x, y, w, h = bbox

        cx = x + w / 2
        cy = y + h / 2

        rows.append({
            "frame": image_map[image_id],
            "image_id": image_id,
            "cx": cx,
            "cy": cy
        })

    df = pd.DataFrame(rows)
    df["cx_heatmap"] = round(df["cx"] / 8, 0)
    df["cy_heatmap"] = round(df["cy"] / 8, 0)

    return df

# ----------------------------
# Build DataFrame (LONG FORMAT)
# ----------------------------
df = load_coco_to_df("train1.json")
df2 = load_coco_to_df("train2.json")
df3 = load_coco_to_df("train3.json")
test = load_coco_to_df("test.json")

dfs = [df, df2, df3, test]

#assume heatmap dimensions are (1920,1080) /8 = (240, 135)

#print(df2.head())


def gaussian_heatmap(h, w, x0, y0, sigma=2):
    """
    Create single Gaussian heatmap.
    """
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)

    heatmap = np.exp(-((xv - x0) ** 2 + (yv - y0) ** 2) / (2 * sigma ** 2))
    return heatmap


def frame_to_heatmap(df_frame, h=135, w=240, sigma=2):
    """
    Convert all points for a single frame into one multi-peak heatmap.
    """
    heatmap = np.zeros((h, w), dtype=np.float32)

    for _, row in df_frame.iterrows():
        x = int(round(row["cx_heatmap"]))
        y = int(round(row["cy_heatmap"]))

        # skip invalid points
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        gauss = gaussian_heatmap(h, w, x, y, sigma)

        # IMPORTANT: max merge (not sum)
        heatmap = np.maximum(heatmap, gauss)

    return heatmap


def dataframe_to_heatmaps(df, h=135, w=240, sigma=2):
    """
    Full dataset conversion: returns dict of frame -> heatmap
    """
    heatmaps = {}

    for frame_name, df_frame in df.groupby("frame"):
        heatmaps[frame_name] = frame_to_heatmap(df_frame, h, w, sigma)

    return heatmaps





#SHOW_HEATMAP
"""
frame_name = "frame_000002.png"
plt.imshow(test[frame_name], cmap="hot")
plt.colorbar()
plt.title(frame_name)
plt.show()
"""


#SHOW_FRAME
"""
video_path = "clip1.mp4"
frame_index = 2  # change this to pick frame

# -----------------------
# OPEN VIDEO
# -----------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

# -----------------------
# SEEK TO FRAME
# -----------------------
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError(f"Could not read frame {frame_index}")

# -----------------------
# CONVERT COLOR (IMPORTANT)
# OpenCV = BGR, Matplotlib = RGB
# -----------------------
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# -----------------------
# DISPLAY
# -----------------------
plt.figure(figsize=(10, 6))
plt.imshow(frame_rgb)
plt.axis("off")
plt.title(f"Frame {frame_index}")
plt.show()
"""
