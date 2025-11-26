import cv2
import random
import mediapipe as mp

VIDEO_PATH = "data/WIN_20251124_13_47_13_Pro.mp4"

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

# Get total frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", total_frames)

# Pick random frame number
random_frame_index = random.randint(0, total_frames - 1)
print("Random frame index:", random_frame_index)

# Jump to that frame
cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame")
    exit()

# Convert to RGB for MediaPipe
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Run FaceMesh
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
            )

# Show the result
cv2.imshow("Random FaceMesh Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
