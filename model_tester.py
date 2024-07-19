import matplotlib.pyplot as plt
import json
import tensorflow as tf
import numpy as np


# load the model that has been packed
def load_model():
    c = np.load('test_X.npy')
    d = np.load('test_y.npy')
    return c, d


# load from file
test_images, test_labels = load_model()

# using following code to load model
loaded_model = tf.keras.models.load_model('my_model.h5')

test_loss, test_acc = loaded_model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy: ', test_acc)

# load history from file
with open('history.json', 'r') as f:
    history_data = json.load(f)


# recreate history object to draw the graph
class History:
    def __init__(self, history):
        self.history = history


history = History(history_data)

# draw the learning curve
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.show()
