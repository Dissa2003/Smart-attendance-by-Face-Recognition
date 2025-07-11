import cv2
import pickle
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsClassifier
import csv
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

# Initialize webcam
video = cv2.VideoCapture(0)

# Load Haar cascade face detector
facedetect = cv2.CascadeClassifier('data/face.xml')

# Load labels and face data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Convert faces to numpy array and flatten
FACES = np.array(FACES).reshape((len(FACES), -1))

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
imgBackground = cv2.imread("background.png")

# Set frame size and position on background
top_offset = 200
left_offset = 180
frame_height = 480
frame_width = 640

required_height = top_offset + frame_height
required_width = left_offset + frame_width

if imgBackground is None or imgBackground.shape[0] < required_height or imgBackground.shape[1] < required_width:
    imgBackground = cv2.resize(imgBackground, (required_width, required_height))

COL_names = ['Name', 'Date', 'Time']

# Start the face recognition loop
while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    attendance = []
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        pred = knn.predict(resized_img)

        current_time = datetime.now()
        date_for_file = current_time.strftime('%Y-%m-%d')  # For filename (safe format)
        time_str = current_time.strftime('%H:%M:%S')       # For display
        datetime_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

        filename = f"Attendence/Attendance_{date_for_file}.csv"
        file_exists = os.path.isfile(filename)

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(pred[0]), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        attendance = [str(pred[0]), date_for_file, time_str]

    # Place frame on background
    imgBackground[top_offset:top_offset+frame_height, left_offset:left_offset+frame_width] = frame
    cv2.imshow('Face Recognition - Attendance System', imgBackground)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('o') and attendance:
        speak("Attendance Taken..")
        time.sleep(1)
        with open(filename, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_names)
            writer.writerow(attendance)

    if key == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
