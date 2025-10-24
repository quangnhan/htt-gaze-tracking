import cv2
import mediapipe as mp
import numpy as np

# 🔹 Replace webcam with video file
VIDEO_PATH = "data/WIN_20251023_22_30_57_Pro.mp4"  # <<-- Change to your video path
cap = cv2.VideoCapture(VIDEO_PATH)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Eye landmarks (based on MediaPipe face mesh indices)
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_EYE_LANDMARKS = [33, 133]   # left eye corners
RIGHT_EYE_LANDMARKS = [362, 263] # right eye corners

def get_landmark_point(landmarks, index, w, h):
    """Convert normalized landmark coordinates to pixel coordinates."""
    return int(landmarks[index].x * w), int(landmarks[index].y * h)

def get_iris_center(landmarks, iris_indices, w, h):
    """Compute the center of iris landmarks."""
    xs = [landmarks[i].x * w for i in iris_indices]
    ys = [landmarks[i].y * h for i in iris_indices]
    return int(np.mean(xs)), int(np.mean(ys))

def eye_gaze_ratio(iris_center, eye_corner_left, eye_corner_right):
    """Compute normalized horizontal ratio of iris within the eye."""
    eye_width = eye_corner_right[0] - eye_corner_left[0]
    if eye_width == 0:
        return 0.5
    return (iris_center[0] - eye_corner_left[0]) / eye_width

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0: 
    fps = 30  # fallback to 30 FPS
delay = int(1000 / fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Get iris centers
        left_iris = get_iris_center(face_landmarks, LEFT_IRIS, w, h)
        right_iris = get_iris_center(face_landmarks, RIGHT_IRIS, w, h)

        # Get eye corners
        left_eye_left = get_landmark_point(face_landmarks, LEFT_EYE_LANDMARKS[0], w, h)
        left_eye_right = get_landmark_point(face_landmarks, LEFT_EYE_LANDMARKS[1], w, h)
        right_eye_left = get_landmark_point(face_landmarks, RIGHT_EYE_LANDMARKS[0], w, h)
        right_eye_right = get_landmark_point(face_landmarks, RIGHT_EYE_LANDMARKS[1], w, h)

        # Compute ratios
        left_ratio = eye_gaze_ratio(left_iris, left_eye_left, left_eye_right)
        right_ratio = eye_gaze_ratio(right_iris, right_eye_left, right_eye_right)
        avg_ratio = (left_ratio + right_ratio) / 2

        # Determine if user is looking at screen
        if 0.4 < avg_ratio < 0.6:
            status = "LOOKING AT SCREEN"
            color = (0, 255, 0)
        else:
            status = "LOOKING AWAY"
            color = (0, 0, 255)

        # === Draw visualization ===

        # Draw iris centers
        cv2.circle(frame, left_iris, 3, color, -1)
        cv2.circle(frame, right_iris, 3, color, -1)

        # Draw iris circles
        left_iris_pts = np.array([get_landmark_point(face_landmarks, i, w, h) for i in LEFT_IRIS], np.int32)
        right_iris_pts = np.array([get_landmark_point(face_landmarks, i, w, h) for i in RIGHT_IRIS], np.int32)
        cv2.polylines(frame, [left_iris_pts], True, (0, 255, 255), 1)
        cv2.polylines(frame, [right_iris_pts], True, (0, 255, 255), 1)

        # Draw eye outlines (between corners)
        cv2.line(frame, left_eye_left, left_eye_right, (255, 255, 0), 1)
        cv2.line(frame, right_eye_left, right_eye_right, (255, 255, 0), 1)

        # Draw circles to mark eye corners
        cv2.circle(frame, left_eye_left, 2, (255, 0, 0), -1)
        cv2.circle(frame, left_eye_right, 2, (255, 0, 0), -1)
        cv2.circle(frame, right_eye_left, 2, (255, 0, 0), -1)
        cv2.circle(frame, right_eye_right, 2, (255, 0, 0), -1)

        # Display status
        cv2.putText(frame, f"{status}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Ratio: {avg_ratio:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Gaze Detection", frame)
    if cv2.waitKey(delay) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
