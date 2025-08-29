# **ASL Translator Studio 👌**

An interactive, AI-powered **American Sign Language (ASL) Translator and Learning Platform** built with **Streamlit**, **YOLO**, and **WebRTC**.
This project allows users to learn and practice ASL letters (A–Z) in real time using their webcam. It also provides **speech-to-text**, **sentence building**, **practice mode**, and a **gamified learning experience**.

---

## 🚀 Features

* 🎙 **Speech to Text**: Convert spoken words to text in multiple languages.
* 🖐 **ASL Detection**: Real-time ASL letter recognition using YOLO and webcam input.
* ✏ **Sentence Builder**: Construct words and sentences by signing letters.
* 🧠 **Practice Mode**: Train with random letters, track accuracy, and get instant feedback.
* 🎮 **Game Mode**: Timed ASL challenge with scoring, attempts tracking, and accuracy metrics.
* 📊 **Streamlit Interface**: User-friendly sidebar navigation with session state management.

---

## 📂 Project Structure

```
.
├── app.py              # Cloud Enviroment Deployment
├── draft.py            # Local Enviroment Deployment
├── requirements.txt    # Python Dependencies
├── asl_detecton.ipynb  # ASL Model Building
├── best-lite.pt        # Pretrained YOLO model for ASL (Included in Repo)
└── README.md           # Project Documentation
```

---

## 🛠 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place your trained YOLO model file (`best-lite.pt`) in the project root directory.

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

The app will start at: [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deployment

This project can run both **locally** and on **Streamlit Cloud**.
To deploy on **Streamlit Cloud**:

1. Push this repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your repo and deploy `app.py`.

> ⚠️ Note: Make sure `requirements.txt` includes all required dependencies and `best-lite.pt` is uploaded or hosted separately.

---

## 📊 Requirements

All dependencies are listed in `requirements.txt`.
Example key dependencies:

* `streamlit`
* `streamlit-webrtc`
* `ultralytics`
* `opencv-python`
* `speechrecognition`
* `av`
* `streamlit-option-menu`

---

## 📖 Lessons & Future Work

* Improve ASL detection accuracy with more robust YOLO training datasets.
* Extend support beyond alphabet letters to full ASL words and sentences.
* Add **Text-to-Speech (TTS)** for signed sentences.
* Explore **multi-user support** for collaborative ASL learning.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License

This project is licensed under the MIT License.

---

👉 Would you like me to also **create the `requirements.txt`** for you (based exactly on your imports) so it’s clean and works on Streamlit Cloud without errors?
