import cv2
from src.gaze_tracker.gaze_tracker import GazeTracking

# Load video instead of webcam
video_path = "data/left_right.mp4"   # <-- change to your video filename
gaze = GazeTracking()
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break   # Stop when video ends

    gaze.refresh(frame)
    annotated = gaze.annotated_frame()
    
    # Get ratios
    horizontal_ratio = gaze._horizontal_ratio()  # 0 (right) to 1 (left)
    
    # Display blink
    if gaze.is_blinking():
        cv2.putText(annotated, "Blinking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Display direction
    print(horizontal := gaze._horizontal_ratio())
    
    # Display all directions at the same time with different colors
    cv2.putText(annotated, "Looking LEFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    # Green
    cv2.putText(annotated, "Looking CENTER", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2) # Cyan/Yellowish
    cv2.putText(annotated, "Looking RIGHT", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)    # Red

    # Optionally, highlight the actual current direction by adding a rectangle
    if gaze.is_left():
        cv2.rectangle(annotated, (40, 85), (300, 120), (0, 255, 0), 2)
    elif gaze.is_center():
        cv2.rectangle(annotated, (40, 135), (350, 170), (255, 255, 0), 2)
    elif gaze.is_right():
        cv2.rectangle(annotated, (40, 185), (300, 220), (0, 0, 255), 2)

    # Display ratios on screen
    cv2.putText(annotated, f"Horizontal ratio: {gaze._horizontal_ratio():.2f}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(annotated, f"Left Horizontal ratio: {gaze.eye_left.gaze_ratio():.2f}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(annotated, f"Right Horizontal ratio: {gaze.eye_right.gaze_ratio():.2f}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    
    cv2.imshow("Gaze Tracker", annotated)

    # Press ESC to exit early
    if cv2.waitKey(33) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
