import cv2
import pickle
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsClassifier

# Initialize video capture from webcam
video = cv2.VideoCapture(0)

# Load Haar cascade face detector
facedetect = cv2.CascadeClassifier('data/face.xml')

# Load labels and face data from pickle files
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Convert faces to numpy array and flatten them for KNN input
FACES = np.array(FACES)
FACES = FACES.reshape((FACES.shape[0], -1))  # (samples, features)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
imgBackground = cv2.imread("background.png")

# Define where to place the webcam frame on the background
top_offset = 200
left_offset = 180
frame_height = 480
frame_width = 640

# Ensure the background is large enough to hold the video frame
required_height = top_offset + frame_height
required_width = left_offset + frame_width

if imgBackground is None or imgBackground.shape[0] < required_height or imgBackground.shape[1] < required_width:
    imgBackground = cv2.resize(imgBackground, (required_width, required_height))

# Start real-time face recognition loop
while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        pred = knn.predict(resized_img)

        # Draw predictions
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(pred[0]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    # Place the frame on the background image
    imgBackground[top_offset:top_offset + frame_height, left_offset:left_offset + frame_width] = frame

    cv2.imshow('Face Recognition - Attendance System', imgBackground)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()