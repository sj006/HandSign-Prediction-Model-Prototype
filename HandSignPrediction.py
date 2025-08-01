import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Try loading the trained model
try:
    model = tf.keras.models.load_model("asl_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    exit()

# ASL Alphabet Mapping
asl_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# FPS calculation
prev_time = 0

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract 21 landmarks, each with x and y (total 42 values)
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

                # Check size before prediction
                if landmarks.shape[0] == 42:
                    prediction = model.predict(landmarks.reshape(1, -1), verbose=0)
                    letter = asl_labels[np.argmax(prediction)]

                    cv2.putText(frame, f'Letter: {letter}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "Invalid landmarks", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # FPS info
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("ASL Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
