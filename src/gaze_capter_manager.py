import os
import time
import cv2
from datetime import datetime

class GazeDatasetRecorder:
    def __init__(self, base_output_dir="output", cooldown=1.0):
        self.base_output_dir = base_output_dir
        self.cooldown = cooldown
        self.last_save_time = 0

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_folder = os.path.join(base_output_dir, f"{timestamp}")
        os.makedirs(self.run_folder, exist_ok=True)

        print(f"[INFO] Dataset folder created: {self.run_folder}")

    def save(self, label, frame):
        """
        Save frame with a label like: 'left' or 'right'
        """
        current_time = time.time()

        if current_time - self.last_save_time < self.cooldown:
            return False

        filename = f"{label}_{int(current_time * 1000)}.jpg"
        filepath = os.path.join(self.run_folder, filename)

        cv2.imwrite(filepath, frame)
        print(f"[SAVED] {filepath}")

        self.last_save_time = current_time
        return True

    def get_run_folder(self):
        return self.run_folder
