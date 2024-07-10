import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

st.set_option('deprecation.showfileUploaderEncoding', False)


# @st.cache(allow_output_mutation=True)
def load_my_model():
    loaded_model = load_model('bt_fun.h5')
    return loaded_model


with st.spinner('loading model into memory......'):
    model = load_my_model()
# print(model.summary())


st.title("Brain-Tumor-Classification")

file = st.file_uploader("Please upload an image of MRI", type=["jpg", "png"])


def import_and_predict(image, model):
    try:
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        return None


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    string = "The provided image contains " + class_names[np.argmax(predictions)]
    st.success(string)
