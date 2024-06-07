from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten
from tensorflow.keras.models import Sequential

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(TimeDistributed(Flatten(), input_shape=input_shape))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
