import time
from tempfile import NamedTemporaryFile

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.title("Cat and Dog Classification")
st.write("Model is loading")
time.sleep(3)
# loading_model()
st.write("Model is loaded")
model = load_model("better_accuracy.h5")
file = st.file_uploader("Upload an Image to predict", type="jpg")
# st.set_option('deprecation.showfileUploaderEncoding', False)
temp_file = NamedTemporaryFile(delete=False)
if file is not None:
    image1 = Image.open(file)
    st.image(image1)
    temp_file.write(file.getvalue())
presssed = st.button("Predict")
if presssed:
    test_image = image.load_img(temp_file.name, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0] <= 0.5:
        st.write("The image classified is cat")
    else:
        st.write("The image classified is dog")
