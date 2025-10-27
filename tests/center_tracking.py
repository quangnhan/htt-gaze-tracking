import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Get the webcam feed
cap = cv2.VideoCapture(1)

# Screen center (adjust for your monitor size and webcam position)
SCREEN_CENTER = (640 // 2, 480 // 2)

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for eyes
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            # Calculate the center of the eyes
            eye_center_x = int((left_eye.x + right_eye.x) * image.shape[1] / 2)
            eye_center_y = int((left_eye.y + right_eye.y) * image.shape[0] / 2)

            # Estimate head pose (simplified)
            # A more robust solution would use the PnP algorithm, but this is a good starting point.
            nose_tip = face_landmarks.landmark[1]
            nose_tip_x = int(nose_tip.x * image.shape[1])
            nose_tip_y = int(nose_tip.y * image.shape[0])
            
            # Draw a circle for the eye center
            cv2.circle(image, (eye_center_x, eye_center_y), 5, (0, 255, 0), -1)
            
            # Determine if looking at screen (simplified logic)
            # Compare the eye center relative to the nose position and screen center.
            # This logic assumes the user is in front of the screen.
            head_yaw = (nose_tip_x - SCREEN_CENTER[0]) / SCREEN_CENTER[0]
            gaze_at_screen = abs(head_yaw) < 0.2 # Threshold for looking straight

            if gaze_at_screen:
                cv2.putText(image, "Looking at screen", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Looking away", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            # Draw face landmarks for visualization
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Display the result
    cv2.imshow('Eye and Gaze Tracking', image)
    
    if cv2.waitKey(5) & 0xFF == 27: # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()