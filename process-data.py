import os
import pandas as pd

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_paths = ['./data/test_alphabet', './data/train_alphabet']

data = []
labels = []
for data_path in data_paths:
    for dir in os.listdir(data_path):
        for img_path in os.listdir(os.path.join(data_path, dir)):
            data_aux = [0] * 42  # Initialize with zeros

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(data_path, dir, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(min(len(hand_landmarks.landmark), 21)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        eps = 1e-7  # division by 0 protection
                        data_aux[i*2] = (x - min(x_))/(max(x_) - min(x_) + eps)
                        data_aux[i*2 + 1] = (y - min(y_))/(max(y_) - min(y_) + eps)

            data.append(data_aux)
            labels.append(dir)

print(len(data))
df = pd.DataFrame(data)
df['label'] = labels

mean_values = df[df != 0].groupby('label').mean()

# fill in errors with mean
for label in df['label'].unique():
    if label != 'Blank':
        df.loc[df['label'] == label] = df.loc[df['label'] == label].replace(0, mean_values.loc[label])

df.to_csv('asl-data.csv', index=False)