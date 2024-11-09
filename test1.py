import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Adjusted MediaPipe parameters for better detection
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.2,  # Lowered confidence threshold
    min_tracking_confidence=0.2
)

DATA_DIR = 'asl_dataset_sign'

# Problematic signs to pay special attention to
problematic_signs = ['6', 'e', 'j', 'm', 'n', 'q', 's', 't']

# Loop through each folder
for dir_ in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(folder_path):
        continue

    for img_path in os.listdir(folder_path)[:1]:
        full_img_path = os.path.join(folder_path, img_path)
        print(f"Processing: {full_img_path}")

        img = cv2.imread(full_img_path)

        if img is not None:
            # Preprocessing steps for better hand detection
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Increase contrast for better detection
            if dir_ in problematic_signs:
                # Apply contrast enhancement
                lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                enhanced = cv2.merge((cl,a,b))
                img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

            # Process the image
            hands_results = hands.process(img_rgb)

            plt.figure(figsize=(5, 5))

            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                plt.title(f'Sign: {dir_} (Hand detected)', color='green')
            else:
                if dir_ in problematic_signs:
                    plt.title(f'Sign: {dir_} (Problem sign - No detection)', color='red')
                else:
                    plt.title(f'Sign: {dir_} (No hand detected)', color='red')
                print(f"No hands detected in: {dir_}")

            plt.imshow(img_rgb)
            plt.axis('off')
        else:
            print(f"Failed to read image: {full_img_path}")

plt.show()

# Print tips for problematic signs
print("\nTips for improving detection of problematic signs:")
print("1. For number '6': Make sure fingers are clearly separated")
print("2. For 'e': Try to capture the hand from a slight angle to show finger position")
print("3. For 'j', 'm', 'n': Keep the hand steady and ensure good lighting")
print("4. For 'q', 's', 't': Ensure clear contrast between fingers")