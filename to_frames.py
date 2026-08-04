import cv2
from pathlib import Path

def extract_frames(video_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.png"), frame)
        frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx} frames to {output_dir}")

if __name__ == "__main__":
    extract_frames("clip 4.mp4", "test_frames/test" )
