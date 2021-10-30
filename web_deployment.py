from random import random
import streamlit as st
import tensorflow as tf
import streamlit as st
import cv2

st.set_page_config(page_title = "Shape Classifier",page_icon="♦")

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model(r'C:\Users\PUsH\Desktop 1\Shape_rec\my_model.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Shape Classification
         """
         )

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (200,200)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.resize(image,(200,200))
        img = preprocessing(img)
        img = img.reshape(1, 200,200, 1)
        prediction = model.predict(img)    
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    classNo = np.argmax(predictions,axis=1)

    if   classNo[0] == 0: label = 'circle'
    elif classNo[0] == 1: label =  'square'
    elif classNo[0] == 2: label =  'star'
    elif classNo[0] == 3: label =  'triangle'

    score = tf.nn.softmax(predictions[0])
    st.success('Detected:{}'.format(label))
    st.write(classNo)
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(label, 100 * np.max(score))
)
