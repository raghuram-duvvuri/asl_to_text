import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

# Dataset folder
DATASET_PATH = "ASL_kaggle"

classes = sorted(os.listdir(DATASET_PATH))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

X = []
y = []

print("Extracting landmarks...")

for label, letter in enumerate(classes):

    folder = os.path.join(DATASET_PATH, letter)
    images = os.listdir(folder)

    for img_name in tqdm(images, desc=f"Processing {letter}"):

        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            landmarks = []

            for lm in hand.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            landmarks = np.array(landmarks)

            # normalize
            landmarks = landmarks.reshape(21,3)
            landmarks = landmarks - landmarks[0]

            max_val = np.max(np.abs(landmarks))
            if max_val != 0:
                landmarks = landmarks / max_val

            landmarks = landmarks.flatten()

            X.append(landmarks)
            y.append(label)

X = np.array(X)
y = np.array(y)

np.save("x_landmarks.npy", X)
np.save("y_labels.npy", y)

print("Done!")
print("Samples:", X.shape)