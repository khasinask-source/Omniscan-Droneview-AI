import streamlit as st
import cv2
import numpy as np

import tempfile
from ultralytics import YOLO

st.title("🚁 Drone Object Detection Demo")

# LOAD MODEL (AUTO DOWNLOAD)

model = YOLO("yolov8n.pt")

option = st.selectbox("Choose Input", ["Image", "Video"])

# -------- IMAGE --------
if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file:
        file_bytes = uploaded_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(frame)
        frame = results[0].plot()

        st.image(frame, channels="BGR")

# -------- VIDEO --------
if option == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

        cap.release()