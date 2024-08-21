import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# import cv2
import urllib
import itertools
import numpy as np
import pandas as Pd
# import seaborn as sns
import random, os, glob
# from imutils import paths
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from urllib.request import urlopen


# libraries that used to turn off warnings.
import warnings
warnings. filterwarnings('ignore')

from keras.preprocessing import image
from keras.utils import to_categorical, load_img,  img_to_array, array_to_img
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from keras.preprocessing.image import ImageDataGenerator

print("TensorFlow version:", tf.__version__)
# Load the model
model = tf.keras.models.load_model('C:/Users/ASUS/PycharmProjects/car_accident_prediction/carpredmodel.h5', compile=False)

# sizing
target_size = (224, 224)

# forecast
def model_testing(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img, dtype=np.uint8)
    img = np.array(img) / 255.0
    p = model.predict(img.reshape(1, 224, 224, 3))
    predicted_class = np.argmax(p[0])

    return img, p, predicted_class


prediction_labels = {0: "damage", 1: "whole"}

# validation -- damage
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0047.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0066.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0070.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0075.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0086.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0104.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0111.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0216.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0227.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0211.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0163.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0003.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0119.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0157.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0217.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0219.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0220.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0223.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0224.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0206.JPEG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/00-damage/0199.JPEG")
# show
plt.imshow(img.squeeze())
plt.title('Maximum Probability: ' + str(np.max(p0[0], axis=-1)) + "\n" + "Predicted class: " + str(prediction_labels[predicted_class]))
plt.show()



# validation -- whole
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0029.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0057.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0071.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0082.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0092.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0097.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0117.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0135.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0201.jpeg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0205.jpeg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0194.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0186.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/data1a/validation/01-whole/0176.jpg")

# validation -- eren
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/cars/damage_1.PNG")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/cars/damage_2.jpeg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/cars/damage_3.jpeg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/araba_verisetleri/cars/whole_1.jpeg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/cars/damage_7.jpeg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/cars/damage_8.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/cars/whole_6.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/cars/whole_7.jpeg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/cars/whole_8.jpg")
img, p0, predicted_class = model_testing("C:/Users/ASUS/Desktop/cars/whole_9.jpeg")


# visualize
plt.imshow(img.squeeze())
plt.title('Maximum Probability: ' + str(np.max(p0[0], axis=-1)) + "\n" + "Predicted class: " + str(prediction_labels[predicted_class]))
plt.show()



"""
img1, p1, predicted_class1 = model_testing('C:/Users/ASUS/Desktop/cars/damage_1.PNG')  # damaged -> 1
img2, p2, predicted_class2 = model_testing('C:/Users/ASUS/Desktop/cars/damage_2.jpeg')  # damaged -> 1
img3, p3, predicted_class3 = model_testing('C:/Users/ASUS/Desktop/cars/whole_1.jpeg')  # whole -> 0
img4, p4, predicted_class4 = model_testing('C:/Users/ASUS/Desktop/cars/whole_2.jpeg')  # whole -> 0


plt.figure(figsize=(20, 60))

# First row of subplots
plt.subplot(141)
plt.axis('off')
plt.imshow(img1.squeeze())
plt.title('Maximum Probability: ' + str(np.max(p1[0], axis=-1)) + "\n" + "Predicted class: " + str(prediction_labels[predicted_class1]))

plt.subplot(142)
plt.axis('off')
plt.imshow(img2.squeeze())
plt.title('Maximum Probability: ' + str(np.max(p2[0], axis=-1)) + "\n" + "Predicted class: " + str(prediction_labels[predicted_class2]))

plt.subplot(143)
plt.axis('off')
plt.imshow(img3.squeeze())
plt.title('Maximum Probability: ' + str(np.max(p3[0], axis=-1)) + "\n" + "Predicted class: " + str(prediction_labels[predicted_class3]))

plt.subplot(144)
plt.axis('off')
plt.imshow(img4.squeeze())
plt.title('Maximum Probability: ' + str(np.max(p4[0], axis=-1)) + "\n" + "Predicted class: " + str(prediction_labels[predicted_class4]))

plt.show()
"""

