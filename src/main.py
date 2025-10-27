import cv2
from .gaze_tracker.face_landmark_detector import FaceLandmarkDetector
from .gaze_tracker.gaze_tracker import GazeTracker

cap = cv2.VideoCapture(1)
detector = FaceLandmarkDetector()
gaze_tracker = GazeTracker()

while cap.isOpened():
    # 1. Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # 2. Detect face landmarks and extract eyes
    left_eye, right_eye = detector.process(frame)
    if left_eye:
        left_eye.draw(frame)
    if right_eye:
        right_eye.draw(frame)

    # 3. Compute gaze direction
    gaze_tracker.set_eyes(left_eye, right_eye)
    is_looking = gaze_tracker.is_looking_at_screen()
    if is_looking:
        cv2.putText(frame, "Looking at Screen", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Not Looking at Screen", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Eyes & Iris (Modular)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
