import cv2

class Iris:
    def __init__(self, center=None, contour=None):
        self.center = center      # (x, y)
        self.contour = contour    # list of (x, y) points, optional

    def draw(self, frame, color=(0, 255, 255)):
        """Draw the iris center and contour if available."""
        if self.center is not None:
            cv2.circle(frame, self.center, 3, color, -1)
        if self.contour is not None:
            cv2.polylines(frame, [self.contour], True, color, 1)
