import cv2


class Eye:
    def __init__(self, left=None, right=None, top=None, bottom=None, iris=None):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.iris = iris

    def get_horizontal_ratio(self):
        """Compute normalized horizontal iris position within the eye."""
        if not (self.left and self.right and self.iris and self.iris.center):
            return 0.5
        eye_width = self.right.x - self.left.x
        if eye_width == 0:
            return 0.5
        return (self.iris.center.x - self.left.x) / eye_width

    def get_vertical_ratio(self):
        """Compute normalized vertical iris position within the eye."""
        if not (self.top and self.bottom and self.iris and self.iris.center):
            return 0.5
        eye_height = self.bottom.y - self.top.y
        if eye_height == 0:
            return 0.5
        return (self.iris.center.y - self.top.y) / eye_height

    def draw(self, frame):
        """Draw eye outline and iris."""
        # Draw horizontal line
        if self.left and self.right:
            cv2.line(
                frame,
                (self.left.x, self.left.y),
                (self.right.x, self.right.y),
                (255, 255, 0),
                1,
            )

        # Draw vertical line
        if self.top and self.bottom:
            cv2.line(
                frame,
                (self.top.x, self.top.y),
                (self.bottom.x, self.bottom.y),
                (0, 255, 255),
                1,
            )

        # Draw iris (center + contour)
        if self.iris:
            self.iris.draw(frame)
