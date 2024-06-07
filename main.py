from src.train import train_model

if __name__ == '__main__':
    video_dir = 'path/to/videos'
    labels = {'class1': 0, 'class2': 1, 'class3': 2}  # Modify as per your classes
    train_model(video_dir, labels)