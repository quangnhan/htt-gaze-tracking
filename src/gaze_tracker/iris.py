import cv2
import numpy as np
from typing import List
from .schema import Point

class Iris:
    def __init__(self, center: Point, contour: List[Point]):
        self.center = center
        self.contour = contour

    def draw(self, frame, color=(0, 255, 255)):
        """Draw the iris center and contour if available."""
        if self.center is not None:
            cv2.circle(frame, (self.center.x, self.center.y), 3, color, -1)

        if self.contour is not None and len(self.contour) > 1:
            contour_array = np.array([[p.x, p.y] for p in self.contour], dtype=np.int32)
            cv2.polylines(frame, [contour_array], True, color, 1)

