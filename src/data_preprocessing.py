import cv2
import os
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

def extract_frames(video_path, frame_rate=1):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            image = cv2.resize(image, (224, 224))
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return frames

def preprocess_frames(frames):
    frames = np.array(frames)
    frames = preprocess_input(frames)
    return frames

def load_data(video_dir, labels, frame_rate=1):
    all_features = []
    all_labels = []
    for label, class_index in labels.items():
        class_dir = os.path.join(video_dir, label)
        for video_name in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_name)
            frames = extract_frames(video_path, frame_rate)
            if len(frames) > 0:
                processed_frames = preprocess_frames(frames)
                all_features.append(processed_frames)
                all_labels.append(class_index)
    return all_features, all_labels