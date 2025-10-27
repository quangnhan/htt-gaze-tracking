import cv2
import mediapipe as mp
import numpy as np
from .eye import Eye
from .iris import Iris
from src.logger import get_logger

logger = get_logger(__name__)

class FaceLandmarkDetector:
    LEFT_IRIS = [469, 470, 471, 472]
    RIGHT_IRIS = [474, 475, 476, 477]
    LEFT_EYE_HORIZONTAL = [33, 133]
    LEFT_EYE_VERTICAL = [159, 145]
    RIGHT_EYE_HORIZONTAL = [362, 263]
    RIGHT_EYE_VERTICAL = [386, 374]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.model = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    def process(self, frame):
        """Return left_eye and right_eye objects with coordinates."""
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark
        return (
            self._create_eye(landmarks, self.LEFT_EYE_HORIZONTAL, self.LEFT_EYE_VERTICAL, self.LEFT_IRIS, w, h),
            self._create_eye(landmarks, self.RIGHT_EYE_HORIZONTAL, self.RIGHT_EYE_VERTICAL, self.RIGHT_IRIS, w, h),
        )

    def _create_eye(self, landmarks, horizontal_eye_indices, vertical_eye_indices, iris_indices, w, h):
        left = self._point(landmarks, horizontal_eye_indices[0], w, h)
        right = self._point(landmarks, horizontal_eye_indices[1], w, h)

        top = self._point(landmarks, vertical_eye_indices[0], w, h) if vertical_eye_indices else None
        bottom = self._point(landmarks, vertical_eye_indices[1], w, h) if vertical_eye_indices else None

        iris_contour = np.array([self._point(landmarks, i, w, h) for i in iris_indices], np.int32)
        iris_center = tuple(np.mean(iris_contour, axis=0).astype(int))
        iris = Iris(center=iris_center, contour=iris_contour)

        return Eye(left=left, right=right, top=top, bottom=bottom, iris=iris)

    @staticmethod
    def _point(landmarks, index, w, h):
        return int(landmarks[index].x * w), int(landmarks[index].y * h)
