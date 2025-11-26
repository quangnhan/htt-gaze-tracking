import cv2
import mediapipe as mp
from .eye import Eye


class GazeTracking:
    """
    Tracks both eyes using MediaPipe FaceMesh + your Eye class.
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # IMPORTANT: enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            use_gpu=True
        )

    def refresh(self, frame):
        """Runs face landmark detection and updates eye objects."""
        self.frame = frame
        h, w = frame.shape[:2]

        results = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            self.eye_left = None
            self.eye_right = None
            return

        landmarks = results.multi_face_landmarks[0].landmark

        # Create Eye objects (your class)
        self.eye_right = Eye(landmarks, w, h, is_right_eye=True)
        self.eye_left = Eye(landmarks, w, h, is_right_eye=False)

    def is_blinking(self):
        """
        Returns True if average blink ratio indicates blinking.
        Your blink ratio = eye_width / eye_height
        - Larger ratio = more closed
        """
        if self.eye_left and self.eye_right:
            blink = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blink > 5.0   # <--- Tune this threshold live
        return False

    def _horizontal_ratio(self):
        if self.eye_left and self.eye_right:
            return (self.eye_left.gaze_ratio() + self.eye_right.gaze_ratio()) / 2

    def is_right(self):
        """User is looking RIGHT (pupil near inner corner)"""
        ratio = self._horizontal_ratio()
        if ratio is not None:
            return ratio < 0.4

    def is_left(self):
        """User is looking LEFT (pupil near outer corner)"""
        ratio = self._horizontal_ratio()
        if ratio is not None:
            return ratio > 0.6

    def is_center(self):
        """User is looking mostly centered"""
        if self.eye_left and self.eye_right:
            return not self.is_left() and not self.is_right()
        
    def annotated_frame(self):
        """Returns frame with eye markers drawn."""
        frame = self.frame.copy()

        if self.eye_left:
            self.eye_left.draw(frame)
        if self.eye_right:
            self.eye_right.draw(frame)

        return frame
