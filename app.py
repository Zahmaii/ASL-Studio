# app.py
import streamlit as st
import cv2
import numpy as np
import time
import random
from ultralytics import YOLO
from streamlit_option_menu import option_menu

# ----------------- APP CONFIG -----------------
st.set_page_config(page_title="ASL Studio", layout="wide")

# ----------------- SIDEBAR MENU -----------------
menu = option_menu(
    menu_title="Main Menu",
    options=["Home", "Live Webcam", "Game", "Picture Mode"],
    icons=["house", "camera-video", "controller", "image"],
    menu_icon="cast",
    default_index=0
)

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    # Replace with your YOLO model path
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# ----------------- HOME PAGE -----------------
if menu == "Home":
    st.title("Welcome to ASL Studio")
    st.write(
        """
        This platform helps you learn and practice American Sign Language (ASL).
        You can use:
        - Live Webcam recognition
        - Picture recognition
        - Fun ASL Games
        """
    )

# ----------------- LIVE WEBCAM PAGE -----------------
elif menu == "Live Webcam":
    st.title("Live Webcam ASL Recognition")

    run_webcam = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access the webcam")
                break

            # Convert frame for YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, stream=False)

            # Example: draw boxes on frame
            annotated_frame = results[0].plot()
            
            # Display in Streamlit
            FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            
            # Allow breaking loop
            if st.button("Stop Webcam"):
                run_webcam = False
                break

        cap.release()
        cv2.destroyAllWindows()

# ----------------- GAME PAGE -----------------
elif menu == "Game":
    st.title("ASL Learning Game")
    st.write("Guess the ASL sign shown in the webcam!")
    signs = ["A", "B", "C", "D", "E"]
    target = random.choice(signs)
    st.write(f"Show the sign: **{target}**")

    # Webcam capture for game
    if st.button("Check Sign"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, stream=False)
            FRAME_WINDOW.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
            st.success("Recognition complete!")
        cap.release()

# ----------------- PICTURE MODE PAGE -----------------
elif menu == "Picture Mode":
    st.title("Picture ASL Recognition")
    uploaded_file = st.file_uploader("Upload a picture", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_img, stream=False)
        annotated_img = results[0].plot()
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
