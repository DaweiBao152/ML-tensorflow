import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Activation
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np
import json


# load the model that has been packed
def load_model():
    a = np.load('train_X.npy')
    b = np.load('train_y.npy')
    return a, b


train_images, train_labels = load_model()

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=train_images.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=32, epochs=3, validation_split=0.3)

# save model as keras file as recommend
model.save('my_model.keras')

# save history as json
with open('history.json', 'w') as f:
    json.dump(history.history, f)
