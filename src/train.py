import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from .data_preprocessing import load_data
from .feature_extraction import extract_features
from .model import build_model

def prepare_data(video_dir, labels, frame_rate=1, max_sequence_length=20):
    all_features, all_labels = load_data(video_dir, labels, frame_rate)
    all_features_extracted = [extract_features(frames) for frames in all_features]
    all_features_padded = pad_sequences(all_features_extracted, maxlen=max_sequence_length, dtype='float32', padding='post')
    num_classes = len(labels)
    all_labels_one_hot = to_categorical(all_labels, num_classes=num_classes)
    return all_features_padded, all_labels_one_hot

def train_model(video_dir, labels, frame_rate=1, max_sequence_length=20, epochs=10, batch_size=8, validation_split=0.2):
    X, y = prepare_data(video_dir, labels, frame_rate, max_sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (max_sequence_length, X.shape[2], X.shape[3], X.shape[4])
    num_classes = len(labels)
    model = build_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return model

if __name__ == '__main__':
    video_dir = 'path/to/videos'
    labels = {'class1': 0, 'class2': 1, 'class3': 2}  # Modify as per your classes
    train_model(video_dir, labels)