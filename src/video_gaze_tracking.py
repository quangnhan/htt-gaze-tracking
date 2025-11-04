import cv2
from src.gaze_tracker.gaze_tracker import GazeTracking

# Load video instead of webcam
video_path = "data/WIN_20251023_22_30_57_Pro.mp4"   # <-- change to your video filename
gaze = GazeTracking()
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break   # Stop when video ends

    gaze.refresh(frame)
    annotated = gaze.annotated_frame()

    # Display blink
    if gaze.is_blinking():
        cv2.putText(annotated, "Blinking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Display direction
    print(horizontal := gaze._horizontal_ratio())
    
    if gaze.is_left():
        cv2.putText(annotated, "Looking LEFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    elif gaze.is_right():
        cv2.putText(annotated, "Looking RIGHT", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(annotated, "Looking CENTER", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gaze Tracker", annotated)

    # Press ESC to exit early
    if cv2.waitKey(33) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
