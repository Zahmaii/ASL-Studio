# streamlit run app.py

import streamlit as st
import cv2
import math
from ultralytics import YOLO
import random
from streamlit_option_menu import option_menu
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import speech_recognition as sr

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
# Custom Audio Processor for Speech-to-Text
# ----------------------
class SpeechToTextProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.text = "Listening..."

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to numpy array
        audio = frame.to_ndarray()
        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.int16)

        # Feed to SpeechRecognition
        try:
            audio_data = sr.AudioData(audio.tobytes(), frame.sample_rate, 2)
            self.text = self.recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            self.text = "Could not understand audio"
        except sr.RequestError:
            self.text = "API unavailable"

        return frame

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
        ["Speech to Text", "ASL Detection", "Practice Mode"],
        icons=["mic", "hand-index-thumb", "clipboard-check"],
        menu_icon="cast",
        default_index=0,
    )

# ----------------------
# SPEECH TO TEXT
# ----------------------
if selected == "Speech to Text":
    st.header("🎙️ Speech to Text (Browser Microphone)")
    
    # Language selection
    lang = st.selectbox("Select language", ("en-EN", "ko-KR", "ja-JP", "zh-CN"))

    # Start WebRTC audio streaming
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=SpeechToTextProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if webrtc_ctx.audio_processor:
        st.subheader("📝 Transcription:")
        st.text(webrtc_ctx.audio_processor.text)

# ----------------------
# ASL DETECTION
# ----------------------
elif selected == "ASL Detection":
    st.header("🖐️ Real-time ASL Detection")
    st.write("Allow access to your webcam below 👇")
    webrtc_streamer(key="asl-detect", video_transformer_factory=ASLTransformer)

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
    st.subheader(f"👉 Try signing this letter: **{st.session_state.target_letter}**")

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
            st.session_state.last_update_time = current_time
