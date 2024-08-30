import shutil
import tkinter as tk
import customtkinter
import tensorflow
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Reshape
from keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

#load the model

path_of_model = input(r"Enter the path of the model\n")
path_of_images = input(r"Enter the path of images\n")
destination_path_Cat = input(r"Enter the path of destination for Cat\n")
destination_path_Dog = input(r"Enter the path of destination for Dog\n")

x = [path_of_model+"\\"+x for x in os.listdir(path_of_model) if x.endswith(".h5")]
pathOfModel = x

model = load_model(pathOfModel[0])

# to get the images
for roots,dirs,files in os.walk(path_of_images):
    for file in files:
        if file.endswith("jpg"):
            img_path = roots+"\\"+file
            test_image = image.load_img(img_path, target_size=(227, 227))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image / 255.0)
            predicted_class = np.argmax(result, axis=1)  # Get the index of the class with the highest probability

            # Assuming class 0 is 'Cat' and class 1 is 'Dog'
            if predicted_class == 0:  # prediction = 'Cat'
                shutil.move(img_path,destination_path_Cat)
            else:  # prediction = 'Dog'
                shutil.move(img_path, destination_path_Dog)



