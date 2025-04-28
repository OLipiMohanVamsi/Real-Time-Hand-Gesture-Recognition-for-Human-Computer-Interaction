# Upload Dataset

from google.colab import files
uploaded = files.upload()

# Unzip the uploaded dataset

import zipfile
import os

zip_path = 'archive(1).zip'  # This is the uploaded file
extract_path = 'gesture_dataset'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"[INFO] Dataset extracted to: {extract_path}")


# Preprocess the dataset (convert videos into frame sequences)

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_dataset(dataset_path, sequence_length=30, img_size=(224, 224)):
    X = []
    y = []
    class_names = sorted(os.listdir(dataset_path))
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for gesture in class_names:
        gesture_dir = os.path.join(dataset_path, gesture)
        if not os.path.isdir(gesture_dir):
            continue
        for video_name in os.listdir(gesture_dir):
            video_path = os.path.join(gesture_dir, video_name)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            if len(frames) >= sequence_length:
                frames = frames[:sequence_length]
                X.append(np.array(frames))
                y.append(label_map[gesture])

    X = np.array(X)
    y = to_categorical(y, num_classes=len(class_names))
    return X, y, class_names

X, y, class_names = preprocess_dataset(extract_path)
print(f"[INFO] Loaded {len(X)} samples across {len(class_names)} classes.")


#  Train the CNN-LSTM Model

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model((30, 224, 224, 3), len(class_names))
model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=8)

#  Save the trained model
model.save("gesture_model.h5")
print("[INFO] Model saved successfully.")

# Real-time Gesture Recognition using the trained  model with webcam
def real_time_prediction():
    import mediapipe as mp
    import tensorflow as tf
    print("[INFO] Starting real-time prediction...")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    sequence = []
    SEQUENCE_LENGTH = 30
    gestures = class_names
    model = tf.keras.models.load_model("gesture_model.h5")

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                sequence.append(np.array(landmarks))
                if len(sequence) == SEQUENCE_LENGTH:
                    input_data = np.array(sequence)
                    input_data = np.expand_dims(input_data, axis=0)
                    prediction = model.predict(input_data)[0]
                    gesture = gestures[np.argmax(prediction)]
                    cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    sequence = []

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


