# streamlit run app.py

import streamlit as st
import cv2
import math
from ultralytics import YOLO
import speech_recognition as sr
import random
from streamlit_option_menu import option_menu
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ASL Class Names
classNames = [chr(i) for i in range(65, 91)]  # A-Z
model = YOLO("best-lite.pt")  # Load once globally


# ----------------------
# Custom Video Transformer for ASL Detection
# ----------------------
class ASLTransformer(VideoTransformerBase):
    def __init__(self):  # <-- fixed from _init_
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

# Title
st.title("ASL Translator Studio 👌")

# Sidebar option menu
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Speech to Text", "ASL Detection", "Practice Mode", "Sentence Builder", "Game Mode"],
        icons=["mic", "hand-index-thumb", "clipboard-check", "align-middle", "controller"],
        menu_icon="cast",
        default_index=0,
    )

# ----------------------
# SPEECH TO TEXT
# ----------------------
if selected == "Speech to Text":
    st.header("🎙 Speech to Text")
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
            st.code(f"⚠ Request Error: {e}", language='txt')

# ----------------------
# ASL DETECTION
# ----------------------
elif selected == "ASL Detection":
    st.header("🖐 Real-time ASL Detection")
    st.write("Allow access to your webcam below 👇")
    webrtc_streamer(key="asl-detect", video_transformer_factory=ASLTransformer)

# ----------------------
# ASL DETECTION (with Sentence Builder)
# ----------------------
if selected == "Sentence Builder":
    st.header("🖐 Real-time ASL Detection with Sentence Builder")
    st.write("Allow access to your webcam below 👇")

    # Initialize sentence state
    if "sentence" not in st.session_state:
        st.session_state.sentence = ""
    if "last_letter_time" not in st.session_state:
        st.session_state.last_letter_time = time.time()

    ctx = webrtc_streamer(key="asl-detect", video_transformer_factory=ASLTransformer)

    # Live-updating placeholder
    sentence_placeholder = st.empty()

    if ctx.video_transformer:
        while ctx.state.playing:   # loop while webcam is active
            detected_letter = ctx.video_transformer.last_letter
            current_time = time.time()

            # Add letter if enough time has passed (avoid duplicate spam)
            if detected_letter and (current_time - st.session_state.last_letter_time) > 2:
                st.session_state.sentence += detected_letter
                st.session_state.last_letter_time = current_time

                # 🔑 reset so the same letter is not spammed
                ctx.video_transformer.last_letter = None

            # Update sentence in UI
            sentence_placeholder.success(st.session_state.sentence)
            time.sleep(0.1)  # short delay to avoid busy loop

    # Show final sentence
    st.subheader("✏ Built Sentence")
    st.success(st.session_state.sentence)

    # 🗑 Clear All button
    if st.button("🗑 Clear All"):
        st.session_state.sentence = ""


# ----------------------
# PRACTICE MODE
# ----------------------
elif selected == "Practice Mode":
    st.header("🧠 ASL Practice Mode (Live)")

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

    # Show target letter
    st.subheader(f"👉 Try signing this letter: *{st.session_state.target_letter}*")

    # Reset button
    if st.button("🔄 Reset Score"):
        st.session_state.score = 0
        st.session_state.attempts = 0

    # Run webcam with ASL Transformer
    ctx = webrtc_streamer(key="asl-practice", video_transformer_factory=ASLTransformer)

    if ctx.video_transformer:
        current_time = time.time()
        detected_letter = ctx.video_transformer.last_letter

        # Every 10 seconds, evaluate the detection and update score
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

            # Update the last update time
            st.session_state.last_update_time = current_time  # ✅ fixed
# ----------------------
# GAME MODE
# ----------------------
elif selected == "Game Mode":
    import streamlit as st
    import subprocess
    import signal
    import os
    
    # Keep process reference
    if "game_process" not in st.session_state:
        st.session_state.game_process = None
    
    st.title("🎮 Game Mode")
    
    # Start button
    if st.button("▶️ Start Game"):
        if st.session_state.game_process is None or st.session_state.game_process.poll() is not None:
            st.session_state.game_process = subprocess.Popen(["python", "game.py"])
            st.success("Game started!")
        else:
            st.warning("Game is already running!")
    
    # Stop button
    if st.button("⏹️ Stop Game"):
        if st.session_state.game_process and st.session_state.game_process.poll() is None:
            os.kill(st.session_state.game_process.pid, signal.SIGTERM)
            st.session_state.game_process = None
            st.success("Game stopped!")
        else:
            st.warning("No game is running.")
