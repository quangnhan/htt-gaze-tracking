import cv2
from src.logger import get_logger
from .eye import Eye

logger = get_logger(__name__)

class GazeTracker:
    def __init__(self):
        self.left_eye = None
        self.right_eye = None
        self.h_threshold = (0.4, 0.6)
        self.v_threshold = (0.4, 0.6)

    def set_h_thresholds(self, h_threshold: tuple):
        self.h_threshold = h_threshold

    def set_v_thresholds(self, v_threshold: tuple):
        self.v_threshold = v_threshold
        
    def set_eyes(self, left_eye: Eye, right_eye: Eye):
        self.left_eye = left_eye
        self.right_eye = right_eye

    def compute_gaze_ratios(self):
        """Return average horizontal and vertical ratios."""
        left_h = self.left_eye.get_horizontal_ratio()
        right_h = self.right_eye.get_horizontal_ratio()
        left_v = self.left_eye.get_vertical_ratio()
        right_v = self.right_eye.get_vertical_ratio()

        return (left_h + right_h) / 2, (left_v + right_v) / 2

    def is_looking_at_screen(self):
        h_ratio, v_ratio = self.compute_gaze_ratios()
        logger.info(f"Gaze Ratios - Horizontal: {h_ratio}, Vertical: {v_ratio}")

        is_center_h = self.h_threshold[0] <= h_ratio <= self.h_threshold[1]
        is_center_v = self.v_threshold[0] <= v_ratio <= self.v_threshold[1]

        return (is_center_h and is_center_v)

    def draw_gaze_box(self, frame):
        """Draw threshold box and current gaze point on frame."""
        if self.left_eye is None or self.right_eye is None:
            return frame

        frame_h, frame_w = frame.shape[:2]

        # Compute gaze ratios
        h_ratio, v_ratio = self.compute_gaze_ratios()

        # Convert ratios to pixel coordinates
        gaze_x = int(h_ratio * frame_w)
        gaze_y = int(v_ratio * frame_h)

        # Convert threshold to pixel rectangle
        x1 = int(self.h_threshold[0] * frame_w)
        x2 = int(self.h_threshold[1] * frame_w)
        y1 = int(self.v_threshold[0] * frame_h)
        y2 = int(self.v_threshold[1] * frame_h)

        # Draw threshold rectangle
        color_box = (0, 255, 0)  # green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)

        # Draw gaze center
        color_point = (0, 0, 255)  # red
        cv2.circle(frame, (gaze_x, gaze_y), 5, color_point, -1)

        return frame