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

# Sidebar option menu
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Speech to Text", "ASL Detection", "Practice Mode", "Game Mode"],
        icons=["mic", "hand-index-thumb", "clipboard-check", "controller"],
        menu_icon="cast",
        default_index=0,
    )

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
    st.header("🖐️ Real-time ASL Detection")
    mode = st.radio("Choose mode:", ["Live", "Picture"])
    if mode == "Live":
        st.write("Allow access to your webcam below 👇")
        webrtc_streamer(key="asl-detect", video_transformer_factory=ASLTransformer)
    else:  # Picture mode
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            results = model(img)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    detected_letter = classNames[cls]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, detected_letter, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Detected: {detected_letter}")

# ----------------------
# PRACTICE MODE
# ----------------------
elif selected == "Practice Mode":
    st.header("🧠 ASL Practice Mode")
    mode = st.radio("Choose mode:", ["Live", "Picture"])
    
    # Session state
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
    if "output_placeholder" not in st.session_state:
        st.session_state.output_placeholder = st.empty()

    st.subheader(f"👉 Try signing this letter: **{st.session_state.target_letter}**")
    if st.button("🔄 Reset Score"):
        st.session_state.score = 0
        st.session_state.attempts = 0

    if mode == "Live":
        ctx = webrtc_streamer(key="asl-practice", video_transformer_factory=ASLTransformer)
        if ctx.video_transformer:
            current_time = time.time()
            detected_letter = ctx.video_transformer.last_letter
            if detected_letter and (current_time - st.session_state.last_update_time) >= 10:
                st.session_state.attempts += 1
                st.session_state.detected_letter = detected_letter
                with st.session_state.output_placeholder.container():
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
    else:  # Picture mode
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            import numpy as np
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            results = model(img)
            detected_letter = None
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    detected_letter = classNames[cls]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, detected_letter, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if detected_letter:
                st.session_state.attempts += 1
                if detected_letter == st.session_state.target_letter:
                    st.session_state.score += 1
                    st.success(f"✅ Correct! You signed: {detected_letter}")
                    st.session_state.target_letter = random.choice(classNames)
                else:
                    st.info(f"✋ Detected: {detected_letter}. Try again.")
                accuracy = (st.session_state.score / st.session_state.attempts) * 100
                st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
                st.write(f"Attempts: {st.session_state.attempts} | Correct: {st.session_state.score}")

# ----------------------
# GAME MODE
# ----------------------
elif selected == "Game Mode":
    st.header("🎮 ASL Game Mode")
    mode = st.radio("Choose mode:", ["Live", "Picture"])
    duration = st.number_input("Game Duration (seconds)", min_value=10, max_value=300, value=30, step=5)

    if "game_start_time" not in st.session_state:
        st.session_state.game_start_time = None
    if "game_score" not in st.session_state:
        st.session_state.game_score = 0
    if "game_letter" not in st.session_state:
        st.session_state.game_letter = random.choice(classNames)
    if "game_last_time" not in st.session_state:
        st.session_state.game_last_time = None
    if "game_output" not in st.session_state:
        st.session_state.game_output = st.empty()

    st.subheader(f"👉 Show this letter: **{st.session_state.game_letter}**")
    if st.button("Start Game"):
        st.session_state.game_start_time = time.time()
        st.session_state.game_last_time = time.time()
        st.session_state.game_score = 0

    if st.session_state.game_start_time:
        elapsed = time.time() - st.session_state.game_start_time
        if elapsed <= duration:
            if mode == "Live":
                ctx = webrtc_streamer(key="game-live", video_transformer_factory=ASLTransformer)
                if ctx.video_transformer:
                    detected_letter = ctx.video_transformer.last_letter
                    current_time = time.time()
                    # Check if letter changes every 5 seconds or if correct
                    if detected_letter:
                        if detected_letter == st.session_state.game_letter or (current_time - st.session_state.game_last_time) >= 5:
                            if detected_letter == st.session_state.game_letter:
                                st.session_state.game_score += 1
                            st.session_state.game_letter = random.choice(classNames)
                            st.session_state.game_last_time = current_time
                            with st.session_state.game_output.container():
                                st.metric("Score", st.session_state.game_score)
                                st.subheader(f"👉 Show this letter: **{st.session_state.game_letter}**")
            else:  # Picture mode
                uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
                if uploaded_file:
                    import numpy as np
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    results = model(img)
                    detected_letter = None
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            detected_letter = classNames[cls]
                    if detected_letter:
                        current_time = time.time()
                        if detected_letter == st.session_state.game_letter or (current_time - st.session_state.game_last_time) >= 5:
                            if detected_letter == st.session_state.game_letter:
                                st.session_state.game_score += 1
                            st.session_state.game_letter = random.choice(classNames)
                            st.session_state.game_last_time = current_time
                            with st.session_state.game_output.container():
                                st.metric("Score", st.session_state.game_score)
                                st.subheader(f"👉 Show this letter: **{st.session_state.game_letter}**")
        else:
            st.success(f"Game over! Your final score: {st.session_state.game_score}")
