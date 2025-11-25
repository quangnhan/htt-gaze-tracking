import math
import cv2
from .pupil import Pupil

class Eye:
    """
    Eye object built on MediaPipe FaceMesh
    """

    def __init__(self, landmarks, image_width, image_height, is_right_eye=True):
        self.landmarks = landmarks
        self.w = image_width
        self.h = image_height
        self.is_right_eye = is_right_eye

        self.pupil = None
        self.blinking = None

        self._analyze()

    def _distance(self, lm1, lm2):
        """Euclidean distance between two MediaPipe landmarks"""
        x1, y1 = lm1.x * self.w, lm1.y * self.h
        x2, y2 = lm2.x * self.w, lm2.y * self.h
        return math.hypot(x2 - x1, y2 - y1)

    def _blink_ratio(self):
        """
        Calculates blink ratio using upper and lower eyelid landmarks.
        Smaller height → blinking / closed eye.
        """
        if self.is_right_eye:
            self.top = self.landmarks[159]
            self.bottom = self.landmarks[145]
            self.left = self.landmarks[33]
            self.right = self.landmarks[133]
        else:
            self.top = self.landmarks[386]
            self.bottom = self.landmarks[374]
            self.left = self.landmarks[362]
            self.right = self.landmarks[263]

        self.eye_height = self._distance(self.top, self.bottom)
        self.eye_width = self._distance(self.left, self.right)

        return self.eye_width / self.eye_height

    def gaze_ratio(self):
        """Returns pupil position ratio from 0.0 (right side) to 1.0 (left side)"""
        pupil_x = self.pupil.x
        left_x = self.left.x * self.w
        right_x = self.right.x * self.w

        eye_width = right_x - left_x
        if eye_width == 0:
            return 0.5  # Prevent divide by zero / fallback

        return (pupil_x - left_x) / eye_width


    def _analyze(self):
        # Create pupil object
        self.pupil = Pupil(
            self.landmarks, 
            self.w, 
            self.h, 
            self.is_right_eye
        )

        # Compute blinking ratio
        self.blinking = self._blink_ratio()

    def draw(self, frame):
        """Draw dots for left, right, top, bottom + pupil"""
        if self.is_right_eye:
            color = (0, 255, 0)  # Green for right eye
        else:
            color = (255, 0, 0)  # Blue for left eye
            
        # Convert landmark to pixel coords
        def px(lm):
            return (int(lm.x * self.w), int(lm.y * self.h))

        # Draw eyelid reference points
        cv2.circle(frame, px(self.left), 3, color, -1)
        cv2.circle(frame, px(self.right), 3, color, -1)
        cv2.circle(frame, px(self.top), 3, color, -1)
        cv2.circle(frame, px(self.bottom), 3, color, -1)

        # Draw pupil
        if self.pupil is not None:
            self.pupil.draw(frame)