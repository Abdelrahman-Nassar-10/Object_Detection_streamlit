import streamlit as st
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image
import numpy as np


st.title("Image Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Analyse Image"):
        with st.spinner('Detecting objects...'):
            bbox, label, conf = cv.detect_common_objects(image_np, confidence=.3)
            
        output_image = draw_bbox(image_np, bbox, label, conf)

        st.image(output_image, caption='Processed Image', use_column_width=True)

        st.subheader("Detected components:")
        if label: 
            
            for idx, item in enumerate(label):
                st.write(f"{idx + 1}. {item}")
            st.success('Object detection completed!')
        else:
            st.warning('No objects detected in the image.')