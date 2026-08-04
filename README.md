# Tennis Ball Detection & Tracking

A heatmap-regression CNN for locating tennis balls in video frames, trained on hand-annotated match footage and built for real-time feasibility.

## Overview

Built as a self-directed project to explore computer vision and to introduce core AI/CV concepts to my lab. The model takes a full-resolution video frame as input and outputs a probability heatmap indicating the ball's location, which is then decoded into precise pixel coordinates.

- **1,325** hand-annotated training frames, **379** test frames
- **100%** precision at a 5px threshold on true-positive detections
- **10.7ms** inference latency on an RTX 3080

## Why Heatmap Regression?

Rather than directly regressing (x, y) coordinates, the model predicts a spatial heatmap and derives coordinates from it. This is deliberate:

- **Direct coordinate regression struggles with small, fast-moving objects.** 
- **No flattening or MLP head required.** A typical CNN flattens its final feature map into a fully-connected head to produce final predictions (e.g. softmax for classification). Since our output of interest is a heatmap or ball centre locations, we can simply use a 1x1 convolutional layer to condense our feature maps into a single map, which our model is trained to finalise as our final heatmap output.
- **Heatmaps give the network a spatial prior.** Ground-truth labels are encoded as Gaussian blobs centered on the ball's true position (see `json_to_heatmap.py`), so nearby pixels share partial "ball-ness" signal. This smooths the signal in loss space.
- **Multi-ball handling via max-merge.** When multiple balls appear in a frame, per-ball Gaussians are combined with an element-wise maximum rather than a sum, preserving distinct peaks instead of blurring them into one blob when balls are close together.

## Architecture

A lightweight CNN (`BallHeatmapCNN` in `model.py`):

- 3 downsampling stages (Conv-BN-ReLU ×2 + MaxPool), reducing spatial resolution by 8× total
- A final feature stage without downsampling
- A 1×1 convolution head producing a single-channel heatmap, passed through a sigmoid

Input: `(B, 3, 1080, 1920)` → Output: `(B, 1, 135, 240)`

Trained with a **weighted MSE loss** that upweights positive (ball) pixels relative to background, since the vast majority of heatmap pixels are near-zero.

## Evaluation

Predicted and ground-truth centres are matched using the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) on pairwise distance, with unmatched predictions/labels counted as false positives/negatives.

- R² (overall, x, y)
- Mean distance error (heatmap-scale and original-resolution)
- PCK@5px / PCK@10px (percentage of correct keypoints within a pixel threshold)
- Precision / recall / false positive / false negative counts

## Results

| Metric | Value |
|---|---|
| Precision @ 5px | 100% |
| False negatives | 0 |
| Inference latency | 10.7ms/frame (RTX 3080) |
| Test frames | 379 |

## AI Usage Disclaimer

Model design, dataset annotation, and evaluation methodology are my own work. Implementation was built with heavy AI assistance, but happy to discuss architecture in further detail upon request.
