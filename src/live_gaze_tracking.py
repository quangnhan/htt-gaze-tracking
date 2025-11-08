import cv2
from src.gaze_tracker.gaze_tracker import GazeTracking

gaze = GazeTracking()
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    gaze.refresh(frame)

    annotated = gaze.annotated_frame()

    if gaze.is_blinking():
        cv2.putText(annotated, "Blinking", (50, 50), 1, 2, (0,0,255), 2)

    # if gaze.is_left():
    #     cv2.putText(annotated, "Looking LEFT", (30, 60), 1, 2, (0,255,0), 2)
    # elif gaze.is_right():
    #     cv2.putText(annotated, "Looking RIGHT", (30, 60), 1, 2, (0,255,0), 2)
    # else:
    #     cv2.putText(annotated, "Looking CENTER", (30, 60), 1, 2, (0,255,0), 2)

    cv2.imshow("Gaze", annotated)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
