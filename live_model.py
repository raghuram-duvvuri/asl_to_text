import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np


classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Model
class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(63,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,26)
        )

    def forward(self,x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ASLModel().to(device)
model.net.load_state_dict(torch.load("asl_landmark_model.pth", map_location=device))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Starting ASL Detection...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            landmarks = np.array(landmarks)

            landmarks = landmarks.reshape(21,3)
            landmarks = landmarks - landmarks[0]

            max_val = np.max(np.abs(landmarks))
            if max_val != 0:
                landmarks = landmarks / max_val

            landmarks = landmarks.flatten()

            tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output,1)
                confidence, predicted = torch.max(prob,1)

            letter = classes[predicted.item()]
            conf = confidence.item()

            cv2.putText(
                frame,
                f"{letter} ({conf:.2f})",
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0,255,0),
                3
            )

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()