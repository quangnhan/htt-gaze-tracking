class GazeCounterFrames:
    def __init__(self):
        self.total_frames = 0
        self.blink_frames = 0
        self.left_frames = 0
        self.right_frames = 0
        self.center_frames = 0

    def update(self, gaze):
        self.total_frames += 1
        if gaze.is_blinking():
            self.blink_frames += 1
        elif gaze.is_left():
            self.left_frames += 1
        elif gaze.is_right():
            self.right_frames += 1
        elif gaze.is_center():
            self.center_frames += 1

    def get_stats(self):
        return {
            "total_time": self.total_frames,
            "blink_time": self.blink_frames,
            "left_time": self.left_frames,
            "right_time": self.right_frames,
            "center_time": self.center_frames,
        }