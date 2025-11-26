import cv2
import tkinter as tk
from PIL import Image, ImageTk

from src.gaze_tracker.gaze_tracker import GazeTracking
from src.gaze_capter_manager import GazeDatasetRecorder
from src.gaze_counter import GazeCounterFrames

VIDEO_PATH = "data/WIN_20251126_11_16_52_Pro.mp4"

# Init systems
gaze = GazeTracking()
recorder = GazeDatasetRecorder(base_output_dir="output", cooldown=1.0)
gaze_counter = GazeCounterFrames()

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Playback state
playing = False
current_frame = 0

# ------------------------- Tkinter UI -------------------------
root = tk.Tk()
root.title("Gaze Tracker Player")

video_label = tk.Label(root)
video_label.pack()

frame_slider = tk.Scale(root,
                        from_=0,
                        to=total_frames,
                        orient=tk.HORIZONTAL,
                        length=900)
frame_slider.pack()

def play():
    global playing
    playing = True

def pause():
    global playing
    playing = False

def slider_changed(value):
    global current_frame
    current_frame = int(value)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

frame_slider.config(command=slider_changed)

controls = tk.Frame(root)
controls.pack(pady=5)

tk.Button(controls, text="▶ Play", width=10, command=play).grid(row=0, column=1)
tk.Button(controls, text="⏸ Pause", width=10, command=pause).grid(row=0, column=2)

status_label = tk.Label(root, text="Frame: 0")
status_label.pack()

# ------------------------- Gaze Dashboard -------------------------
def draw_gaze_dashboard(frame, gaze):
    options = ["BLINK", "LEFT", "CENTER", "RIGHT"]

    start_x = 20
    start_y = 100
    box_width = 160
    box_height = 50
    spacing = 10

    inactive_color = (120, 120, 120)

    colors = {
        "BLINK": (0, 0, 255),
        "LEFT": (0, 255, 0),
        "CENTER": (255, 255, 0),
        "RIGHT": (0, 165, 255)
    }

    if gaze.is_blinking():
        active = "BLINK"
    elif gaze.is_left():
        active = "LEFT"
    elif gaze.is_center():
        active = "CENTER"
    elif gaze.is_right():
        active = "RIGHT"
    else:
        active = None

    for i, option in enumerate(options):
        x = start_x
        y = start_y + i * (box_height + spacing)

        color = colors[option] if option == active else inactive_color

        cv2.rectangle(frame, (x, y),
                      (x + box_width, y + box_height),
                      color, -1)

        cv2.putText(frame,
                    option,
                    (x + 15, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2)

    return active

# ------------------------- Update Loop -------------------------
def update_frame():
    global current_frame

    if playing:
        ret, frame = cap.read()
        if not ret:
            return

        current_frame += 1

        gaze.refresh(frame)
        gaze_counter.update(gaze)
        annotated = gaze.annotated_frame()

        # Frame counter
        cv2.putText(annotated,
                    f"Frame: {current_frame}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Draw gaze UI dashboard
        active = draw_gaze_dashboard(annotated, gaze)

        # Save logic only when highlighted
        if active == "LEFT":
            recorder.save("left", annotated)
        elif active == "RIGHT":
            recorder.save("right", annotated)

        # Convert for Tkinter display
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)

        frame_slider.set(current_frame)
        status_label.config(text=f"Frame: {current_frame}/{total_frames}")

    root.after(1, update_frame)

# ------------------------- Start -------------------------
update_frame()
root.mainloop()

cap.release()

print(gaze_counter.get_stats())
