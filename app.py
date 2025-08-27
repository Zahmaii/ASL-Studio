# app.py
import streamlit as st
import cv2
import math
import time
import random
import os
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import urllib.request
from PIL import Image
import numpy as np
import speech_recognition as sr

# ----------------------
# Ensure YOLO model exists
# ----------------------
MODEL_PATH = "best-lite.pt"
MODEL_URL = "https://your-storage.com/best-lite.pt"  # <-- replace with your actual URL if using cloud

if not os.path.exists(MODEL_PATH):
    st.warning("Downloading YOLO model, please wait...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded!")

# Load YOLO model
model = YOLO(MODEL_PATH)

# ASL Class Names
classNames = [chr(i) for i in range(65, 91)]  # A-Z

# ----------------------
# Custom Video Transformer
# ----------------------
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_letter = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                detected_letter = classNames[cls]

                # Draw bounding box + label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, detected_letter, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.last_letter = detected_letter

        return img

# ----------------------
# Streamlit Page Config
# ----------------------
st.set_page_config(
    page_title="ASL Translator Studio",
    page_icon="👌",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("ASL Translator Studio 👌")

# ----------------------
# Sidebar Menu
# ----------------------
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Speech to Text", "ASL Detection", "Practice Mode", "Game Mode"],
        icons=["mic", "hand-index-thumb", "clipboard-check", "controller"],
        menu_icon="cast",
        default_index=0,
    )

    # Choose mode (Live or Picture)
    mode_choice = st.radio("Mode", ["Live", "Picture"], index=0)

# ----------------------
# SPEECH TO TEXT
# ----------------------
if selected == "Speech to Text":
    st.header("🎙️ Speech to Text")
    lang = st.selectbox("Select language", ("en-EN", "ko-KR", "ja-JP", "zh-CN"))
    if st.button("Recognize"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            notice = st.text("Say something...")
            speech = r.listen(source)
        try:
            audio = r.recognize_google(speech, language=lang)
            notice.empty()
            st.code(audio, language='txt')
        except sr.UnknownValueError:
            notice.empty()
            st.code("❌ Could not understand your speech.", language='txt')
        except sr.RequestError as e:
            notice.empty()
            st.code(f"⚠️ Request Error: {e}", language='txt')

# ----------------------
# ASL DETECTION
# ----------------------
elif selected == "ASL Detection":
    st.header("🖐️ ASL Detection")

    if mode_choice == "Live":
        st.write("Allow access to your webcam below 👇")
        webrtc_streamer(key="asl-detect", video_transformer_factory=ASLTransformer)
    else:
        uploaded_file = st.file_uploader("Upload a picture for ASL detection", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(img)
            results = model(img_array)
            detected_letters = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    detected_letters.append(classNames[cls])
            st.image(img, caption="Uploaded Image", use_column_width=True)
            if detected_letters:
                st.success(f"Detected letters: {', '.join(detected_letters)}")
            else:
                st.info("No letters detected.")

# ----------------------
# PRACTICE MODE
# ----------------------
elif selected == "Practice Mode":
    st.header("🧠 ASL Practice Mode")

    # Session state setup
    if "target_letter" not in st.session_state:
        st.session_state.target_letter = random.choice(classNames)
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "attempts" not in st.session_state:
        st.session_state.attempts = 0
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = time.time()
    if "detected_letter" not in st.session_state:
        st.session_state.detected_letter = None

    st.subheader(f"👉 Try signing this letter: **{st.session_state.target_letter}**")

    if st.button("🔄 Reset Score"):
        st.session_state.score = 0
        st.session_state.attempts = 0

    if mode_choice == "Live":
        ctx = webrtc_streamer(key="practice-live", video_transformer_factory=ASLTransformer)
        if ctx.video_transformer:
            current_time = time.time()
            detected_letter = ctx.video_transformer.last_letter
            if detected_letter and (current_time - st.session_state.last_update_time) >= 10:
                st.session_state.attempts += 1
                st.session_state.detected_letter = detected_letter

                if detected_letter == st.session_state.target_letter:
                    st.session_state.score += 1
                    st.success(f"✅ Correct! You signed: {detected_letter}")
                    st.session_state.target_letter = random.choice(classNames)
                else:
                    st.info(f"✋ Detected: {detected_letter}. Try again.")

                accuracy = (st.session_state.score / st.session_state.attempts) * 100
                st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
                st.write(f"Attempts: {st.session_state.attempts} | Correct: {st.session_state.score}")
                st.session_state.last_update_time = current_time
    else:
        uploaded_file = st.file_uploader("Upload a picture to practice", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(img)
            results = model(img_array)
            detected_letters = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    detected_letters.append(classNames[cls])
            if detected_letters:
                detected_letter = detected_letters[0]
                st.success(f"Detected letter: {detected_letter}")
                st.session_state.attempts += 1
                if detected_letter == st.session_state.target_letter:
                    st.session_state.score += 1
                    st.session_state.target_letter = random.choice(classNames)
                accuracy = (st.session_state.score / st.session_state.attempts) * 100
                st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
                st.write(f"Attempts: {st.session_state.attempts} | Correct: {st.session_state.score}")

# ----------------------
# GAME MODE
# ----------------------
elif selected == "Game Mode":
    st.header("🎮 ASL Game Mode")

    # Game settings
    if "game_score" not in st.session_state:
        st.session_state.game_score = 0
    if "game_letter" not in st.session_state:
        st.session_state.game_letter = random.choice(classNames)
    if "game_last_time" not in st.session_state:
        st.session_state.game_last_time = time.time()
    if "game_start_time" not in st.session_state:
        st.session_state.game_start_time = time.time()
    if "game_duration" not in st.session_state:
        st.session_state.game_duration = 30
    if "game_detected_letter" not in st.session_state:
        st.session_state.game_detected_letter = None

    game_time = st.number_input("Game Duration (seconds)", min_value=10, max_value=120, value=30, step=5)
    st.session_state.game_duration = game_time

    st.subheader(f"Current Letter: **{st.session_state.game_letter}**")
    st.write(f"Score: {st.session_state.game_score}")

    if mode_choice == "Live":
        ctx = webrtc_streamer(key="game-live", video_transformer_factory=ASLTransformer)
        if ctx.video_transformer:
            current_time = time.time()
            detected_letter = ctx.video_transformer.last_letter
            elapsed_time = current_time - st.session_state.game_start_time

            # Game over
            if elapsed_time > st.session_state.game_duration:
                st.success(f"Game Over! Final Score: {st.session_state.game_score}")
            else:
                if detected_letter and (current_time - st.session_state.game_last_time) >= 0:
                    st.session_state.game_detected_letter = detected_letter
                    if detected_letter == st.session_state.game_letter:
                        st.session_state.game_score += 1
                        st.session_state.game_letter = random.choice(classNames)
                        st.session_state.game_last_time = current_time
                    elif (current_time - st.session_state.game_last_time) >= 5:
                        st.session_state.game_letter = random.choice(classNames)
                        st.session_state.game_last_time = current_time
    else:
        uploaded_file = st.file_uploader("Upload a picture for the game", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(img)
            results = model(img_array)
            detected_letters = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    detected_letters.append(classNames[cls])
            if detected_letters:
                detected_letter = detected_letters[0]
                st.session_state.game_detected_letter = detected_letter
                if detected_letter == st.session_state.game_letter:
                    st.session_state.game_score += 1
                    st.session_state.game_letter = random.choice(classNames)
                    st.session_state.game_last_time = time.time()
