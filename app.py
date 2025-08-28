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

# Title
# st.title("ASL Translator Studio 👌")

# Sidebar option menu
with st.sidebar:
    selected = option_menu(
        "ASL Translator Studio 👌",
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
# SENTENCE BUILDER
# ----------------------
if selected == "Sentence Builder":
    st.header("🖐 Real-time ASL Detection with Sentence Builder")
    st.write("Allow access to your webcam below 👇")

    if "sentence" not in st.session_state:
        st.session_state.sentence = ""
    if "last_letter_time" not in st.session_state:
        st.session_state.last_letter_time = time.time()

    ctx = webrtc_streamer(key="asl-sentence", video_transformer_factory=ASLTransformer)
    sentence_placeholder = st.empty()

    if ctx.video_transformer:
        while ctx.state.playing:
            detected_letter = ctx.video_transformer.last_letter
            current_time = time.time()

            if detected_letter and (current_time - st.session_state.last_letter_time) > 2:
                st.session_state.sentence += detected_letter
                st.session_state.last_letter_time = current_time
                ctx.video_transformer.last_letter = None

            sentence_placeholder.success(st.session_state.sentence)
            time.sleep(0.1)

    st.subheader("✏ Built Sentence")
    st.success(st.session_state.sentence)

    if st.button("🗑 Clear All"):
        st.session_state.sentence = ""


# ----------------------
# PRACTICE MODE
# ----------------------
elif selected == "Practice Mode":
    st.header("🧠 ASL Practice Mode (Live)")

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

    st.subheader(f"👉 Try signing this letter: *{st.session_state.target_letter}*")

    if st.button("🔄 Reset Score"):
        st.session_state.score = 0
        st.session_state.attempts = 0

    ctx = webrtc_streamer(key="asl-practice", video_transformer_factory=ASLTransformer)

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


# ----------------------
# GAME MODE (Streamlit-native)
# ----------------------
elif selected == "Game Mode":
    st.header("🎮 ASL Letter Game")

    # Setup session state
    if "game_running" not in st.session_state:
        st.session_state.game_running = False
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "attempts" not in st.session_state:
        st.session_state.attempts = 0
    if "target_letter" not in st.session_state:
        st.session_state.target_letter = random.choice(classNames)
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "game_duration" not in st.session_state:
        st.session_state.game_duration = 30
    if "last_letter_time" not in st.session_state:
        st.session_state.last_letter_time = time.time()

    # Game duration selector
    st.session_state.game_duration = st.slider("⏱ Select game duration (seconds)", 10, 120, 30)

    # Start / Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Game"):
            st.session_state.game_running = True
            st.session_state.score = 0
            st.session_state.attempts = 0
            st.session_state.start_time = time.time()
            st.session_state.target_letter = random.choice(classNames)
            st.success("Game started! Show the letters with your hand ✋")
    with col2:
        if st.button("⏹️ Stop Game"):
            st.session_state.game_running = False
            st.success("Game stopped!")

    if st.session_state.game_running:
        elapsed = time.time() - st.session_state.start_time
        remaining = st.session_state.game_duration - elapsed

        if remaining <= 0:
            st.session_state.game_running = False
            st.subheader("⏱ Time's up!")
            accuracy = (st.session_state.score / st.session_state.attempts) * 100 if st.session_state.attempts > 0 else 0
            st.success(f"✅ Final Score: {st.session_state.score}")
            st.info(f"🎯 Accuracy: {accuracy:.1f}%")
        else:
            st.write(f"⏳ Time left: {int(remaining)} seconds")
            st.subheader(f"👉 Sign this letter: **{st.session_state.target_letter}**")

            ctx = webrtc_streamer(key="asl-game", video_transformer_factory=ASLTransformer)

            if ctx.video_transformer:
                detected_letter = ctx.video_transformer.last_letter
                current_time = time.time()

                # Correct detection or 5 seconds timeout
                if detected_letter:
                    if detected_letter == st.session_state.target_letter:
                        st.session_state.score += 1
                        st.session_state.attempts += 1
                        st.success(f"✅ Correct! You signed {detected_letter}")
                        st.session_state.target_letter = random.choice(classNames)
                        st.session_state.last_letter_time = current_time
                        ctx.video_transformer.last_letter = None
                    elif (current_time - st.session_state.last_letter_time) >= 5:
                        st.session_state.attempts += 1
                        st.warning(f"⌛ Time up for {st.session_state.target_letter}. New letter!")
                        st.session_state.target_letter = random.choice(classNames)
                        st.session_state.last_letter_time = current_time
                        ctx.video_transformer.last_letter = None

            st.metric("🏆 Score", st.session_state.score)
            st.metric("📊 Attempts", st.session_state.attempts)

