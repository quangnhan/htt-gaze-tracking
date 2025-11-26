import cv2
import time
from src.gaze_tracker.gaze_tracker import GazeTracking
from src.gaze_counter import GazeCounterFrames

VIDEO_PATH = "data/WIN_20251126_11_16_52_Pro.mp4"

# Init systems
gaze = GazeTracking()
gaze_counter = GazeCounterFrames()

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

# Start timing
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_start = time.time()
    current_frame += 1

    # Process gaze
    gaze.refresh(frame)
    gaze_counter.update(gaze)

    frame_end = time.time()
    frame_time = frame_end - frame_start

    # Print progress
    print(f"Processed frame {current_frame}/{total_frames} ({frame_time:.3f} sec)")

cap.release()

# Total processing time
total_time = time.time() - start_time
print(f"\nTotal processing time: {total_time:.2f} sec")
print(f"Average time per frame: {total_time/current_frame:.3f} sec")

# Print final gaze statistics
print(gaze_counter.get_stats())
