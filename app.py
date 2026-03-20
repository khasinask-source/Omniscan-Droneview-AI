import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")
st.title("🚁 OmniScan Drone AI")
st.subheader("Smart Object Detection & Counting System")

# ---------------- MODEL ----------------
model = YOLO("yolov8m.pt")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.3)

option = st.selectbox("Choose Input", ["Image", "Video"])

# ================= IMAGE =================
if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file:
        file_bytes = uploaded_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        start = time.time()
        results = model(frame, conf=confidence)
        end = time.time()

        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int) if boxes else []

        names = model.names
        count_dict = {}

        for class_id in class_ids:
            class_name = names[class_id]
            count_dict[class_name] = count_dict.get(class_name, 0) + 1

        frame = results[0].plot()

        st.image(frame, channels="BGR")

        # Sidebar output
        st.sidebar.write("### Object Counts")
        for k, v in count_dict.items():
            st.sidebar.write(f"{k}: {v}")

        fps = 1 / (end - start)
        st.sidebar.write(f"FPS: {fps:.2f}")

        # Download
        cv2.imwrite("output.jpg", frame)
        with open("output.jpg", "rb") as file:
            st.download_button("Download Result", file, file_name="result.jpg")

# ================= VIDEO =================
elif option == "Video":
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

            start = time.time()
            results = model(frame, conf=confidence)
            end = time.time()

            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int) if boxes else []

            names = model.names
            count_dict = {}

            for class_id in class_ids:
                class_name = names[class_id]
                count_dict[class_name] = count_dict.get(class_name, 0) + 1

            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

            # Sidebar updates
            st.sidebar.write("### Object Counts")
            for k, v in count_dict.items():
                st.sidebar.write(f"{k}: {v}")

            fps = 1 / (end - start)
            st.sidebar.write(f"FPS: {fps:.2f}")

        cap.release()
