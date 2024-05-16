import os
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import mediapipe as mp

# Your existing setup for mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the data paths
data_paths = ['./data/test_alphabet'] # ['./data/test_alphabet', './data/train_alphabet']

# Initialize the data and labels list
original_data = []
augmented_data = []
labels = []

# Define the ImageDataGenerator for augmentation
data_gen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

# Function to augment images
def augment_image(img):
    img = img.reshape((1,) + img.shape)  # Reshape image
    it = data_gen.flow(img, batch_size=1)  # Create a generator
    return next(it)[0].astype('uint8')  # Get the next image from generator

# Function to process and extract features from an image
def process_image(img_rgb):
    results = hands.process(img_rgb)
    data_aux = [0] * 63  # Initialize with zeros
    x_ = []
    y_ = []
    z_ = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                x_.append(x)
                y_.append(y)
                z_.append(z)

            for i in range(min(len(hand_landmarks.landmark), 21)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                eps = 1e-7  # division by 0 protection
                data_aux[i*3] = (x - min(x_))/(max(x_) - min(x_) + eps)
                data_aux[i*3 + 1] = (y - min(y_))/(max(y_) - min(y_) + eps)
                data_aux[i*3 + 2] = (z - min(z_)) / (max(z_) - min(z_) + eps)

    return data_aux

for data_path in data_paths:
    for dir in os.listdir(data_path):
        for img_path in os.listdir(os.path.join(data_path, dir)):
            img = cv2.imread(os.path.join(data_path, dir, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            labels.append(dir)

            original_features = process_image(img_rgb)
            original_data.append(original_features)

            augmented_img_rgb = augment_image(img_rgb)
            augmented_features = process_image(augmented_img_rgb)
            augmented_data.append(augmented_features)

original_df = pd.DataFrame(original_data)
original_df['label'] = labels

augmented_df = pd.DataFrame(augmented_data)
augmented_df['label'] = labels

median_values = original_df[original_df != 0].groupby('label').median()

mask = (original_df.loc[:, original_df.columns != 'label'] == 0).all(axis=1)

original_df.loc[mask, 'label'] = 'Blank'

original_df.loc[original_df['label'] == 'Blank', original_df.columns != 'label'] = 0.0
augmented_df.loc[augmented_df['label'] == 'Blank', original_df.columns != 'label'] = 0.0

# Save the DataFrames to CSV
original_df.to_csv('./data/asl-datav3.csv', index=False)
augmented_df.to_csv('./data/asl-datav3-augmented.csv', index=False)
