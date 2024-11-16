
## This is the driving code for the emotion recognition part of the application using numpy,opencv and keras.
## The model has been trained on the 'fer2023' dataset from kaggle.
## Its should be noted that the model is not 100% accurate and any slight change in facial expression results in a 
## completely different emotion as perceived by the model. To solve this problem, the camera window is opened
## for only 5 seconds and the most common emotion recorded in that time frame is used for any further steps.

import tensorflow
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import time

USE_WEBCAM = True
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
frame_window = 10
emotion_offsets = (20, 40)

face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_window = []

cv2.namedWindow('window_frame')

video_capture = cv2.VideoCapture(0)  # 0 for default camera or source selection

cap = None
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)  # Webcam source
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4')  # Video file source

start_time = time.time()
duration = 5  # recording duration in seconds

while cap.isOpened() and (time.time() - start_time) < duration:
    ret, bgr_image = cap.read()

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                           minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Exit when 'Q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()

# Determine the most common facial expression in the recorded window
final_emotion = mode(emotion_window)
final_emotion=final_emotion.capitalize()
if final_emotion in ["Angry", "Surprise"]:
    final_emotion = "Energetic"



    
from select_music import music_select
from play_music import play_youtube_song

sel_song=music_select(final_emotion)
print(f"Playing {sel_song} because you're looking kinda {final_emotion}")
play_youtube_song(sel_song)
 
