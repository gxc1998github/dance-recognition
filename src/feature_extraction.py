from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

def extract_features(frames):
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    features = model.predict(frames)
    return features