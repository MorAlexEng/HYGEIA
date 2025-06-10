'''
__title__: A remote health monitoring platform to enhance health monitoring in home-based healthcare using Raspberry pi, OpenCV, Keras and Tensorflow
           (Modified to use robust DNN-based face, age, and gender detection)

__author__ : Alex Mor
'''

import cv2
import time
import datetime
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time as t
import csv
import os
from tensorflow.keras.models import load_model

def play_video(video):
    video_cap = cv2.VideoCapture(video)
    while True:
        video_ret, video_frame = video_cap.read()
        if not video_ret:
            break
        cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Video', video_frame)
        if cv2.waitKey(25) & 0xFF == ord('s'):
            break
    cv2.destroyWindow('Video')

# ---------------------------
# Face, Age & Gender Detection Setup
# ---------------------------
# Use the constants and model files from your age_gender_detection.py for robust detection.

FACE_PROTO = "constants/opencv_face_detector.pbtxt"
FACE_MODEL = "constants/opencv_face_detector_uint8.pb"
AGE_PROTO = "constants/age_deploy.prototxt"
AGE_MODEL = "constants/age_net.caffemodel"
GENDER_PROTO = "constants/gender_deploy.prototxt"
GENDER_MODEL = "constants/gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


# Load networks

faceNet = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)


# Modified getFaceBox: converts 4-channel images to 3-channel

def getFaceBox(net, frame, conf_threshold=0.7):
    # If the image has 4 channels (BGRA), convert it to BGR
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes


# ---------------------------
# Load the pre-trained emotion detection model
# ---------------------------

model = load_model('fer_model.h5')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']


# Demonstration videos

intro_video_path = '/home/mpl/heartpi/videos/emotion.mp4'
videos = ['/home/mpl/heartpi/videos/healthy.mp4',
          '/home/mpl/heartpi/videos/tachycardia.mp4',
          '/home/mpl/heartpi/videos/vradycardia.mp4',
          '/home/mpl/heartpi/videos/danger.mp4']

# ---------------------------
# Camera initialization
# ---------------------------

from picamera2 import Picamera2
picam2 = Picamera2()
picam2.start()
time.sleep(0.1)  # Allow the camera to warm up



# ---------------------------
# Variables for heart rate measurement
# ---------------------------

firstFrame = None
start = None
time_list = []   # To store timestamps (in seconds)
R = []
G = []
B = []
frame_num = 0
plt.ion()
median_counter_max = 50
median_counter = 0
temp_median = 0



# Create CSV file (filename includes current timestamp)

start_time_csv = datetime.datetime.now()
csv_filename = f"file_{start_time_csv}.csv"
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['time', 'Gender', 'age', 'bpm', 'median', 'emotion', 'confidence'])

frames_with_no_faces = 0
intro_flag = True

cv2.namedWindow("ROI", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("ROI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



# ---------------------------
# Main Loop
# ---------------------------

while True:
    frame = picam2.capture_array()
    if frame.size == 0:
        print("Empty Frame")
        time.sleep(1)
        continue

    # Use robust DNN-based face detection
    frame_face, bboxes = getFaceBox(faceNet, frame)
    
    if len(bboxes) == 0:
        cv2.putText(frame, "No Face Detected!", (int(frame.shape[1] / 2) - 100, int(frame.shape[0] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow('ROI', frame)
        firstFrame = None
        frames_with_no_faces += 1
        if frames_with_no_faces >= 50:
            frame_num = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        continue
    else:
        frames_with_no_faces = 0

    # On the first valid detection, initialize timing and perform age/gender detection on all faces for annotation
    if firstFrame is None:
        start = datetime.datetime.now()
        time_list.append(0)
        firstFrame = frame.copy()
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            padding = 20
            face_roi_color = frame[max(0, y1 - padding):min(y2 + padding, frame.shape[0] - 1),
                                   max(0, x1 - padding):min(x2 + padding, frame.shape[1] - 1)]
            face_roi_color = cv2.resize(face_roi_color, (227, 227))
            blob = cv2.dnn.blobFromImage(face_roi_color, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            detected_gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            detected_age = ageList[agePreds[0].argmax()]
            print(f'Gender: {detected_gender}')
            print(f'Age: {detected_age[1:-1]} years')
            cv2.putText(frame_face, f"{detected_gender}, {detected_age}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Update time and frame count
    current_time = (datetime.datetime.now() - start).total_seconds()
    time_list.append(current_time)
    frame_num += 1

    # Use the first detected face (bbox) for emotion and heart rate measurement
    x1, y1, x2, y2 = bboxes[0]
    padding = 20

    # Emotion detection (on grayscale face ROI)
    face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_roi_emotion = face_gray[y1:y2, x1:x2]
    try:
        face_roi_emotion = cv2.resize(face_roi_emotion, (48, 48))
    except Exception as e:
        continue
    face_roi_emotion = face_roi_emotion / 255.0
    face_roi_emotion = np.expand_dims(face_roi_emotion, axis=0)
    predictions = model.predict(face_roi_emotion)
    predicted_emotion = np.argmax(predictions)
    confidence_score = round(predictions[0][predicted_emotion], 4) * 100

    cv2.putText(frame_face, emotions[predicted_emotion], (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame_face, f"{int(confidence_score)}%", (x2 - 50, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Heart rate measurement: compute average color over the face region
    ROI_mask = np.zeros_like(frame)
    ROI_mask = cv2.rectangle(ROI_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    ROI_mask_gray = cv2.cvtColor(ROI_mask, cv2.COLOR_BGR2GRAY)
    ROI_color = cv2.bitwise_and(frame, frame, mask=ROI_mask_gray)
    R_new, G_new, B_new, _ = cv2.mean(ROI_color, mask=ROI_mask_gray)
    R.append(R_new)
    G.append(G_new)
    B.append(B_new)

    cv2.imshow('ROI', frame_face)

    if frame_num < 900:
        cv2.putText(frame_face, f"Please stay still   {int(frame_num / 9)}%", (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        N = 900
        # Handle intro video playback
        if intro_flag:
            intro_video = cv2.VideoCapture(intro_video_path)
            intro_flag = False
            intro_video_ret, intro_video_frame = intro_video.read()
        if intro_video_ret:
            cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Video', intro_video_frame)
            intro_video_ret, intro_video_frame = intro_video.read()
        else:
            intro_flag = True
            # Heart rate analysis via FFT
            G_std = StandardScaler().fit_transform(np.array(G[-(N - 1):]).reshape(-1, 1)).flatten()
            R_std = StandardScaler().fit_transform(np.array(R[-(N - 1):]).reshape(-1, 1)).flatten()
            B_std = StandardScaler().fit_transform(np.array(B[-(N - 1):]).reshape(-1, 1)).flatten()
            T = 1 / (len(time_list[-(N - 1):]) / (time_list[-1] - time_list[-(N - 1)]))
            X_f = FastICA(n_components=3).fit_transform(np.array([R_std, G_std, B_std]).T).T
            N_fft = len(X_f[0])
            yf = fft(X_f[1])
            yf = yf / np.sqrt(N_fft)
            xf = fftfreq(N_fft, T)
            xf = fftshift(xf)
            yplot = fftshift(abs(yf))
            fft_plot = yplot.copy()
            fft_plot[xf <= 0.75] = 0
            # Consider frequencies between 0 and 4 Hz; multiply by 60 for BPM
            bpm_value = xf[(xf >= 0) & (xf <= 4)][fft_plot[(xf >= 0) & (xf <= 4)].argmax()] * 60
            data = str(bpm_value) + ' bpm'
            print(data)
            # Accumulate median BPM values
            if median_counter < median_counter_max:
                median_counter += 1
                temp_median += bpm_value
            else:
                median_num = temp_median / median_counter
                median_str = str(median_num) + ' bpm'
                median_counter = 0
                temp_median = 0
                # Record data in CSV
                with open(csv_filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([datetime.datetime.now(), detected_gender, detected_age, data,
                                     median_str, emotions[predicted_emotion], confidence_score])
                # Video playback based on median BPM
                if median_num > 120:
                    play_video(videos[3])
                    print('1')
                elif 80 < median_num < 120:
                    play_video(videos[1])
                    print('2')
                elif 40 < median_num < 80:
                    play_video(videos[0])
                    print('3')
                elif 20 < median_num < 40:
                    play_video(videos[2])
                else:
                    play_video(videos[3])
                    print('4')
                firstFrame = None
                frame_num = 0
                intro_flag = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
