import streamlit as st
import requests
import cv2
import numpy as np
import os
import base64
from io import BytesIO
import pandas as pd

st.title("Brain Tumor Segmentation App")
tab1, tab2 = st.tabs(["Home","Results"])
df = pd.read_csv("files/log.csv")


with tab1:
    st.write("Please Input an MRI scan Here")
    images = st.file_uploader("Select an MRI image", type=["png","jpg"], accept_multiple_files=False, key=None, label_visibility="visible")
    if images is not None:
        save_image_path = os.path.join("Dataset/Brain_MRI/images/", images.name)
        st.image(images)
    segment = st.button("Segment Image")
    if segment:
        byte_data = images.read()
        response = requests.post("https://brain-tumor.herokuapp.com/brain-tumor", files={"file": ("image.png", byte_data, 'image/png')})
        if response.status_code == 200:
            decoded_array = response.content
            decoded_array = np.frombuffer(decoded_array, np.uint8)
            decoded_image = cv2.imdecode(decoded_array, cv2.IMREAD_COLOR)
            st.image(decoded_image)


with tab2:
    st.dataframe(df)           