# 🩺 Skin Disease Detector

An AI-powered web application that detects common skin conditions from an uploaded image using a Convolutional Neural Network (CNN) built on MobileNetV2, wrapped in an interactive, animated Streamlit interface.

---

## 📖 Overview

Skin Disease Detector is a deep learning project that classifies an uploaded skin image into one of six common skin conditions. It uses a fine-tuned MobileNetV2 CNN for image classification and presents the results through a modern, professional, dark/light-themed dashboard — complete with an animated confidence gauge, probability chart, symptom cards, and treatment recommendations.

This project was built as an academic/AI project to demonstrate an end-to-end Supervised Learning $\to$ Multi-Class Image Classification pipeline, from model training to a fully deployed, user-facing web app.

---

## ✨ Features

* Upload a skin image (JPG/PNG) directly from the browser
* AI-powered prediction using a fine-tuned MobileNetV2 CNN
* Animated confidence gauge showing prediction certainty
* Probability bar chart comparing all 6 disease classes
* Detailed disease description, symptoms, and treatment suggestions
* Dark Mode / Light Mode toggle with a fully themed UI
* Modern glassmorphism-style cards with smooth animations
* Clean, responsive, professional dashboard layout

---

## 🧬 Detected Conditions

| Class | Description |
| --- | --- |
| **Acne** | Clogged hair follicles causing pimples, blackheads, and cysts |
| **Eczema** | Chronic inflammation causing dry, itchy, cracked skin |
| **Psoriasis** | Autoimmune disorder causing thick, scaly red patches |
| **Rosacea** | Chronic facial redness and visible blood vessels |
| **Vitiligo** | Loss of skin pigment causing white patches |
| **Warts** | Small skin growths caused by HPV infection |

---

## 🏗️ Model & Machine Learning Details

| Aspect | Detail |
| --- | --- |
| Learning Type | Supervised Learning |
| Task Type | Multi-Class Image Classification |
| Base Architecture | MobileNetV2 (pre-trained on ImageNet, fine-tuned) |
| Output Layer | Softmax (6-class probability distribution) |
| Input Size | $224 \times 224$ RGB |
| Framework | TensorFlow / Keras |
| Model Format | `.keras` |
| Training Callbacks | ModelCheckpoint, EarlyStopping, ReduceLROnPlateau |

---

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Web Framework:** Streamlit
* **Image Processing:** OpenCV, NumPy
* **Visualization:** Matplotlib
* **Report Generation:** ReportLab

---

## 📂 Project Structure

```text
Skin-Disease-Detector/
│
├── app/
│   └── app.py                # Main Streamlit application
│
├── models/
│   └── skin_detector_best_7cls.keras   # Trained model (see note below)
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore

```

> **Note:** The dataset and/or trained model files may be excluded from this repository due to GitHub's file size limits. See Model & Dataset Access below.

---

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/zainabfatima2979/Skin-Disease-Detector.git
cd Skin-Disease-Detector

```


2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

```


3. **Install dependencies**
```bash
pip install -r requirements.txt

```


4. **Add the trained model**
Place the trained `.keras` model file in the path expected by `app.py` (see `MODEL_PATH` variable), or update that path to match your local setup.
5. **Run the app**
```bash
streamlit run app/app.py

```


6. Open the local URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## 📦 Model & Dataset Access

The trained model file may be too large for GitHub. If it has been excluded from this repository, you can download it here:

* 🔗 **Model download link:** *(add your Google Drive / Hugging Face link here)*
* 🔗 **Dataset source:** *(add dataset source/link here)*

---

## ⚠️ Disclaimer

This tool is developed for educational and academic purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for any skin-related concerns.

---

## 🚀 Future Improvements

* Grad-CAM heatmap visualization for model explainability
* PDF report generation for predictions
* Top-3 prediction display with confidence breakdown
* User authentication and prediction history
* Multi-model ensemble for improved accuracy

---

## 👤 Author

**Zainab Fatima**

Feel free to connect or reach out with feedback and suggestions.

---

## 📄 License

This project is licensed under the MIT License.