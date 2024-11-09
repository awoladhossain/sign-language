import os
import mediapipe as mp
import cv2
import pickle

# Data collection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = 'asl_dataset_sign'
data = []
labels = []

# Loop through each folder in the dataset directory
for dir_ in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, dir_)

    # Skip if not a directory
    if not os.path.isdir(folder_path):
        continue

    # Loop through each image in the folder
    for img_path in os.listdir(folder_path):
        data_aux = []
        full_img_path = os.path.join(folder_path, img_path)
        print(f"Reading image: {full_img_path}")  # Debugging output

        # Read the image
        img = cv2.imread(full_img_path)

        # Check if the image was read successfully
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hands_results = hands.process(img_rgb)

            # Ensure hands were detected
            if hands_results.multi_hand_landmarks is None:
                print(f"No hands detected in image: {full_img_path}")
                continue

            # Extract the hand landmarks
            for hand_landmarks in hands_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(dir_)

# Save the collected data and labels
with open("american.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print(f"Data collection complete! Collected {len(data)} samples.")