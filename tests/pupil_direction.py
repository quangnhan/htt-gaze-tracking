import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # THIS enables iris landmarks
)

# Eye corner indices (from FaceMesh)
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Iris center index
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        mesh = res.multi_face_landmarks[0].landmark
        
        # Get coordinates
        left_eye_left = np.array([mesh[LEFT_EYE_LEFT].x * w, mesh[LEFT_EYE_LEFT].y * h])
        left_eye_right = np.array([mesh[LEFT_EYE_RIGHT].x * w, mesh[LEFT_EYE_RIGHT].y * h])
        left_iris = np.array([mesh[LEFT_IRIS_CENTER].x * w, mesh[LEFT_IRIS_CENTER].y * h])

        # Compute horizontal ratio (0 → left, 1 → right)
        left_ratio = np.linalg.norm(left_iris - left_eye_left) / np.linalg.norm(left_eye_right - left_eye_left)

        # Display direction
        if left_ratio < 0.4:
            direction = "Looking LEFT"
        elif left_ratio > 0.6:
            direction = "Looking RIGHT"
        else:
            direction = "Looking CENTER"

        cv2.putText(frame, direction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        cv2.circle(frame, tuple(left_iris.astype(int)), 3, (0,255,255), -1)

    cv2.imshow("Pupil Direction", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
