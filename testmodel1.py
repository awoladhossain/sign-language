# Real-time prediction
import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained model
model_dict = pickle.load(open("americansign.p", "rb"))
model = model_dict["model"]
print(f"Number of features expected by the model: {model.n_features_in_}")

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    data_aux = []

    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there's an issue capturing frames

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract 84 features (x and y for each of the 21 landmarks)
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.append(x)
                data_aux.append(y)

        # Ensure the data has the expected number of features (84)
        if len(data_aux) < 84:
            data_aux = np.pad(data_aux, (0, 84 - len(data_aux)), 'constant')
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]

        data_aux = np.array([data_aux])  # Reshape for the model
        predictions = model.predict(data_aux)
        print(f"Predictions: {predictions}")
    else:
        print("Warning: No hands detected in the frame.")

    cv2.imshow('frame', frame)

    # Allow exiting the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()