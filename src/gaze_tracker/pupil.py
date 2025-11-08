import cv2

class Pupil:
    """
    Detects the pupil position using MediaPipe FaceMesh (468 landmarks).
    """

    def __init__(self, landmarks, image_width, image_height, is_right_eye=True):
        """
        Arguments:
            landmarks: List of 468 MediaPipe face landmarks
            image_width: Width of the image frame
            image_height: Height of the image frame
            is_right_eye: True -> right pupil (landmark 468), False -> left pupil (landmark 473)
        """
        self.landmarks = landmarks
        self.w = image_width
        self.h = image_height
        self.is_right_eye = is_right_eye

        self.x = None
        self.y = None

        self.detect_pupil()

    def detect_pupil(self):
        # Iris center landmarks from MediaPipe FaceMesh
        iris_center_index = 468 if self.is_right_eye else 473

        try:
            lm = self.landmarks[iris_center_index]
            # Convert from normalized (0–1) to pixel coordinates
            self.x = int(lm.x * self.w)
            self.y = int(lm.y * self.h)
        except:
            self.x, self.y = None, None

    def draw(self, frame):
        """
        Draws the detected pupil on the frame.

        Arguments:
            frame (numpy.ndarray): Frame to draw on
        """
        color = (0, 0, 255)
        if self.x is not None and self.y is not None:
            cv2.circle(frame, (self.x, self.y), 4, color, -1)
            