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
        eye_width = self.right[0] - self.left[0]
        if eye_width == 0:
            return 0.5
        return (self.iris.center[0] - self.left[0]) / eye_width

    def get_vertical_ratio(self):
        """Compute normalized vertical iris position within the eye."""
        if not (self.top and self.bottom and self.iris and self.iris.center):
            return 0.5
        eye_height = self.bottom[1] - self.top[1]
        if eye_height == 0:
            return 0.5
        return (self.iris.center[1] - self.top[1]) / eye_height

    def draw(self, frame):
        """Draw eye outline and iris."""
        if self.left and self.right:
            cv2.line(frame, self.left, self.right, (255, 255, 0), 1)
        if self.top and self.bottom:
            cv2.line(frame, self.top, self.bottom, (0, 255, 255), 1)
        if self.iris:
            self.iris.draw(frame)
