import cv2

video_path = "data/WIN_20251126_11_16_52_Pro.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception(f"Cannot open video: {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    raise Exception("FPS is 0. Video cannot be read or codec not supported.")

duration = total_frames / fps

print(f"Total frames: {total_frames}")
print(f"FPS: {fps}")
print(f"Duration: {duration:.2f} seconds")
