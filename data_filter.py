import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
import random
from pathlib import Path

# store training data
DATADIR = r'D:\project\machine_learning\machine_learning_start\.venv\Scripts\new\PetImages'
# store test data
TESTDIR = r'D:\project\machine_learning\machine_learning_start\.venv\Scripts\new\Testing'

CATEGORIES = ['Dog', 'Cat']

"""
# check if data load correctly (only load first image)
# load image in to array
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for image in os.listdir(path):
        img_array = (cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE))
        plt.imshow(img_array, cmap='gray')
        print(os.path.join(path, image))
        plt.show()
        break
    break

# get the size
print(img_array.shape)

IMG_SIZE = 64
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
"""


# create training dataset
def create_training_data(dir):
    IMG_SIZE = 128
    arr = []
    for category in CATEGORIES:
        path = os.path.join(dir, category)
        class_num = CATEGORIES.index(category)
        for image in os.listdir(path):
            try:
                img_array = (cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE))
                if img_array is None:
                    print("Error: Image not loaded correctly")
                    break

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                arr.append([new_array, class_num])
            except Exception as e:
                print(e)
                pass
    return arr


def create_test_data(dir):
    IMG_SIZE = 128
    arr = []
    for category in CATEGORIES:
        path = os.path.join(dir, category)
        class_num = CATEGORIES.index(category)
        for image in os.listdir(path):
            try:
                img_array = (cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE))
                if img_array is None:
                    print("Error: Image not loaded correctly")
                    break

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                arr.append([new_array, class_num])
            except Exception as e:
                print(e)
                pass
    return arr


training_data = create_training_data(DATADIR)
testing_data = create_test_data(TESTDIR)

random.shuffle(training_data)
random.shuffle(testing_data)

"""
# random select few training data to see correctness
for sample in training_data[:10]:
    plt.imshow(sample[0], cmap='gray')
    plt.show()
    print(sample[1])

# random select few training data to see correctness
for sample in testing_data[:10]:
    plt.imshow(sample[0], cmap='gray')
    plt.show()
    print(sample[1])
"""

train_X = []
train_y = []
test_X = []
test_y = []

for features, label in training_data:
    train_X.append(features)
    train_y.append(label)

for features, label in testing_data:
    test_X.append(features)
    test_y.append(label)

# should be same image data above
IMG_SIZE = 128
train_X = np.array(train_X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_X = np.array(test_X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# save .npy file
# this will create 4 .npy file, each contain the data required
np.save('train_X.npy', train_X)
np.save('test_X.npy', test_X)
np.save('train_y.npy', train_y)
np.save('test_y.npy', test_y)
