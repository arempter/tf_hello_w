from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# image manipulation
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#from PIL import Image

print(tf.__version__)

data_labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',  4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


def load_model(path):
    model = keras.models.load_model(path)
    return model
    

def load_image(filename):
  # load the image
  img = load_img(filename, target_size=(32, 32))
  # convert to array
  img = img_to_array(img)
  # reshape into a single sample with 3 channels
  img = img.reshape(1, 32, 32, 3)
  # prepare pixel data
  img = img.astype('float32')
  img = img / 255.0
  return img  


def predict(model, img):
    image = load_image(img)
    print(image.shape)

    preditions = model.predict(image)
    print(preditions[0])
    label = np.argmax(preditions[0])
    print("predicted label {}".format(label))
    return data_labels.get(label)
