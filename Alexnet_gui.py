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
#Create root object from cutomtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("1200*700")
root.title("Alexnet Model")

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=40,padx=60,fill="both",expand=True)

label = customtkinter.CTkLabel(master=frame,text="Alexnel Model Cat and Dog",font=("Roboto",24))
label.pack(pady=12,padx=10)

modelPath = customtkinter.CTkEntry(master=frame,text_color="white",placeholder_text="Model Path")
modelPath.pack(pady=12,padx=10)

imagePath = customtkinter.CTkEntry(master=frame,text_color="white",placeholder_text="Image Path")
imagePath.pack(pady=12,padx=10)
prediction = ''


def toTestImg():
    model_path = modelPath.get()
    image_path = imagePath.get()
    model =''
    for roots,dirs,files in os.walk(model_path):
        for f in files:
            if f.endswith(".h5"):
                model_path_ = roots+"\\"+f
                model = load_model(model_path_)
    for roots,dirs,files in os.walk(image_path):
        for f in files:
            if f.endswith(".jpg"):
                img_path = ""
                img_path = roots+"\\"+f
                test_image = image.load_img(img_path,target_size=(227,227))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image,axis=0)
                result = model.predict(test_image/255.0)
                predicted_class = np.argmax(result, axis=1)  # Get the index of the class with the highest probability

                # Assuming class 0 is 'Cat' and class 1 is 'Dog'
                if predicted_class == 0:
                    prediction = 'Cat'
                else:
                    prediction = 'Dog'

            resultImg = customtkinter.CTkEntry(master=frame,text_color="white",placeholder_text=prediction)
            resultImg.pack(pady=12,padx=10)


image_test = customtkinter.CTkButton(master=root,text="test the input image",command=toTestImg)
image_test.pack(pady=12,padx=10)
root.mainloop()