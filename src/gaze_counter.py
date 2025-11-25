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

    def get_stats_only(self):
        return {
            "total_time": self.total_frames,
            "blink_time": self.blink_frames,
            "left_time": self.left_frames,
            "right_time": self.right_frames,
            "center_time": self.center_frames,
        }

    def get_stats(self, frame_delay_ms):
        sec_per_frame = frame_delay_ms / 1000
        return {
            "total_time": round(self.total_frames * sec_per_frame, 2),
            "blink_time": round(self.blink_frames * sec_per_frame, 2),
            "left_time": round(self.left_frames * sec_per_frame, 2),
            "right_time": round(self.right_frames * sec_per_frame, 2),
            "center_time": round(self.center_frames * sec_per_frame, 2),
        }
