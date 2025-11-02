import cv2
from .gaze_tracker.face_landmark_detector import FaceLandmarkDetector
from .gaze_tracker.gaze_tracker import GazeTracker

cap = cv2.VideoCapture(1)
detector = FaceLandmarkDetector()
gaze_tracker = GazeTracker()

# --- Create Trackbars (0 to 100 → later scaled to 0.0–1.0) ---
cv2.namedWindow("Eyes & Iris (Modular)")
cv2.createTrackbar("H_Min", "Eyes & Iris (Modular)", 40, 100, lambda x: None)
cv2.createTrackbar("H_Max", "Eyes & Iris (Modular)", 60, 100, lambda x: None)
cv2.createTrackbar("V_Min", "Eyes & Iris (Modular)", 40, 100, lambda x: None)
cv2.createTrackbar("V_Max", "Eyes & Iris (Modular)", 60, 100, lambda x: None)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Read threshold values from trackbars (scale back to 0.0–1.0)
    h_min = cv2.getTrackbarPos("H_Min", "Eyes & Iris (Modular)") / 100
    h_max = cv2.getTrackbarPos("H_Max", "Eyes & Iris (Modular)") / 100
    v_min = cv2.getTrackbarPos("V_Min", "Eyes & Iris (Modular)") / 100
    v_max = cv2.getTrackbarPos("V_Max", "Eyes & Iris (Modular)") / 100

    # Detect eyes
    left_eye, right_eye = detector.process(frame)
    if left_eye: left_eye.draw(frame)
    if right_eye: right_eye.draw(frame)

    gaze_tracker.set_eyes(left_eye, right_eye)
    gaze_tracker.draw_gaze_box(frame)

    # Use dynamic threshold values
    gaze_tracker.set_h_thresholds((h_min, h_max))
    gaze_tracker.set_v_thresholds((v_min, v_max))
    is_looking = gaze_tracker.is_looking_at_screen()

    label = "Looking at Screen" if is_looking else "Not Looking at Screen"
    color = (0, 255, 0) if is_looking else (0, 0, 255)
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Eyes & Iris (Modular)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
