import cv2
from .gaze_tracker.face_landmark_detector import FaceLandmarkDetector
from .gaze_tracker.gaze_tracker import GazeTracker

# --- Replace webcam with video file ---
VIDEO_PATH = "data/WIN_20251023_22_30_57_Pro.mp4"  # <<< your video file
cap = cv2.VideoCapture(VIDEO_PATH)

detector = FaceLandmarkDetector()
gaze_tracker = GazeTracker()

# --- FPS-based playback ---
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30
delay = int(1000 / fps)

# --- Create Trackbars (0–100) ---
cv2.namedWindow("Eyes & Iris (Modular)")
cv2.createTrackbar("H_Min", "Eyes & Iris (Modular)", 40, 100, lambda x: None)
cv2.createTrackbar("H_Max", "Eyes & Iris (Modular)", 60, 100, lambda x: None)
cv2.createTrackbar("V_Min", "Eyes & Iris (Modular)", 40, 100, lambda x: None)
cv2.createTrackbar("V_Max", "Eyes & Iris (Modular)", 60, 100, lambda x: None)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert trackbars → normalized threshold range
    h_min = cv2.getTrackbarPos("H_Min", "Eyes & Iris (Modular)") / 100
    h_max = cv2.getTrackbarPos("H_Max", "Eyes & Iris (Modular)") / 100
    v_min = cv2.getTrackbarPos("V_Min", "Eyes & Iris (Modular)") / 100
    v_max = cv2.getTrackbarPos("V_Max", "Eyes & Iris (Modular)") / 100

    # Detect eyes
    left_eye, right_eye = detector.process(frame)

    if left_eye: 
        left_eye.draw(frame)
    if right_eye: 
        right_eye.draw(frame)

    # Send eyes to gaze tracker
    gaze_tracker.set_eyes(left_eye, right_eye)

    # Set dynamic thresholds
    gaze_tracker.set_h_thresholds((h_min, h_max))
    gaze_tracker.set_v_thresholds((v_min, v_max))

    # Draw gaze threshold box & gaze point
    gaze_tracker.draw_gaze_box(frame)

    # Determine if user is looking at the screen
    is_looking = gaze_tracker.is_looking_at_screen()
    label = "Looking at Screen" if is_looking else "Not Looking at Screen"
    color = (0, 255, 0) if is_looking else (0, 0, 255)

    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Eyes & Iris (Modular)", frame)
    if cv2.waitKey(delay) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
