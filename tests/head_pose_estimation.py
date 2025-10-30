import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    static_image_mode=False
)

# FaceMesh landmark IDs used for head pose
LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 226,
    "right_eye_outer": 446,
    "mouth_left": 57,
    "mouth_right": 287
}

# 3D model reference points (generic human head model)
MODEL_POINTS = np.array([
    [0.0,   0.0,   0.0],    # Nose tip
    [0.0,  -63.6, -12.5],   # Chin
    [-43.3, 32.7, -26.0],   # Left eye outer
    [43.3,  32.7, -26.0],   # Right eye outer
    [-28.9,-28.9,-24.1],    # Left mouth corner
    [28.9, -28.9,-24.1],    # Right mouth corner
], dtype=float)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # Get 2D image points from FaceMesh
        IMAGE_POINTS = np.array([
            [lm[LANDMARK_IDS["nose_tip"]].x * w, lm[LANDMARK_IDS["nose_tip"]].y * h],
            [lm[LANDMARK_IDS["chin"]].x * w, lm[LANDMARK_IDS["chin"]].y * h],
            [lm[LANDMARK_IDS["left_eye_outer"]].x * w, lm[LANDMARK_IDS["left_eye_outer"]].y * h],
            [lm[LANDMARK_IDS["right_eye_outer"]].x * w, lm[LANDMARK_IDS["right_eye_outer"]].y * h],
            [lm[LANDMARK_IDS["mouth_left"]].x * w, lm[LANDMARK_IDS["mouth_left"]].y * h],
            [lm[LANDMARK_IDS["mouth_right"]].x * w, lm[LANDMARK_IDS["mouth_right"]].y * h]
        ], dtype=float)

        # Camera matrix
        focal_length = w
        CENTER = (w / 2, h / 2)
        CAMERA_MATRIX = np.array([
            [focal_length, 0, CENTER[0]],
            [0, focal_length, CENTER[1]],
            [0, 0, 1]
        ], dtype=float)

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP → rvec, tvec
        _, rvec, tvec = cv2.solvePnP(MODEL_POINTS, IMAGE_POINTS, CAMERA_MATRIX, dist_coeffs)

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Convert to Euler angles (yaw, pitch, roll)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        angles = np.array(angles)  # FIX: convert tuple → numpy array
        yaw, pitch, roll = angles * 180.0 / np.pi

        # Draw orientation line from nose tip
        nose_3d_target = np.array([[0, 0, 100]], dtype=float)
        nose_2d_target, _ = cv2.projectPoints(nose_3d_target, rvec, tvec, CAMERA_MATRIX, dist_coeffs)

        p1 = (int(IMAGE_POINTS[0][0]), int(IMAGE_POINTS[0][1]))
        p2 = (int(nose_2d_target[0][0][0]), int(nose_2d_target[0][0][1]))

        cv2.line(frame, p1, p2, (0, 255, 0), 3)

        # Display results
        cv2.putText(frame, f"Yaw   : {yaw:.1f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
        cv2.putText(frame, f"Pitch : {pitch:.1f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
        cv2.putText(frame, f"Roll  : {roll:.1f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)

    cv2.imshow("Head Pose 3D", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
