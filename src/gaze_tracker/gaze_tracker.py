from src.logger import get_logger

logger = get_logger(__name__)

class GazeTracker:
    def __init__(self):
        self.left_eye = None
        self.right_eye = None

    def set_eyes(self, left_eye, right_eye):
        self.left_eye = left_eye
        self.right_eye = right_eye

    def compute_gaze_ratios(self):
        """Return average horizontal and vertical ratios."""
        left_h = self.left_eye.get_horizontal_ratio()
        right_h = self.right_eye.get_horizontal_ratio()
        left_v = self.left_eye.get_vertical_ratio()
        right_v = self.right_eye.get_vertical_ratio()

        return (left_h + right_h) / 2, (left_v + right_v) / 2

    def is_looking_at_screen(self, h_threshold=(0.4, 0.6), v_threshold=(0.4, 0.6)):
        h_ratio, v_ratio = self.compute_gaze_ratios()
        logger.info(f"Gaze Ratios - Horizontal: {h_ratio}, Vertical: {v_ratio}")

        is_center_h = h_threshold[0] <= h_ratio <= h_threshold[1]
        is_center_v = v_threshold[0] <= v_ratio <= v_threshold[1]

        return (is_center_h and is_center_v)