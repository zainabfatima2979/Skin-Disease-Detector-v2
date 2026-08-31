import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Skin Disease Detector",
    layout="wide"
)

# ================= THEME SELECT =================
st.sidebar.markdown("### 🎨 Select Theme")
theme = st.sidebar.radio("", ["Dark Mode", "Light Mode"])

if theme == "Dark Mode":
    BG = "#0b0f19"
    CARD = "linear-gradient(145deg, #020617, #1f2937)"
    TEXT = "#e5e7eb"
    SUBTEXT = "#cbd5f5"
    TITLE = "#60a5fa"
    GLOW = "rgba(56,189,248,0.45)"
else:
    BG = "#f8fafc"
    CARD = "#ffffff"
    TEXT = "#0f172a"
    SUBTEXT = "#334155"
    TITLE = "#1e3a8a"
    GLOW = "rgba(37,99,235,0.25)"

# ================= FULL PAGE CSS =================
st.markdown(f"""
<style>
html, body, .stApp {{
    background-color: {BG} !important;
    color: {TEXT} !important;
}}

header, section.main > div {{
    background-color: {BG} !important;
}}

label {{
    color: {TEXT} !important;
    font-weight: 600;
}}

h1 {{
    color: {TITLE} !important;
    text-align: center;
    font-size: 44px;
}}

h2 {{
    color: {TITLE} !important;
    text-align: center;
}}

div[data-testid="stFileUploader"] {{
    background: {CARD} !important;
    border-radius: 22px !important;
    padding: 20px !important;
    box-shadow: 0 0 30px {GLOW};
}}

.result-box {{
    background: {CARD};
    padding: 32px;
    border-radius: 24px;
    margin-top: 30px;
    box-shadow: 0 0 30px {GLOW};
    text-align: center;
}}

.disease {{
    font-size: 38px;
    font-weight: bold;
    color: #22d3ee;
}}

.percent {{
    font-size: 26px;
    color: #22c55e;
}}

.treatment-box {{
    background: {CARD};
    padding:16px;
    margin:12px auto;
    width:45%;
    border-radius:16px;
    color:{TEXT};
    text-align:center;
    box-shadow:0 0 18px rgba(34,197,94,0.4);
}}

.description-box {{
    background: {CARD};
    padding:22px;
    margin:20px auto;
    width:70%;
    border-radius:18px;
    color:{TEXT};
    text-align:center;
    box-shadow:0 0 22px rgba(147,197,253,0.35);
    font-size:16px;
}}

footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("<h1>🩺 Skin Disease Detector</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:center; color:{SUBTEXT};'>Upload skin image → AI predicts disease</p>",
    unsafe_allow_html=True
)

# ================= LOAD MODEL =================
MODEL_PATH = r"C:\Users\user\Desktop\Skin Detector\archive\models\skin_detector_best_6cls.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Acne", "Eczema", "Psoriasis", "Rosacea", "Vitiligo", "Warts"]
IMG_SIZE = 224

# ================= TREATMENTS =================
recommendations = {
    "Acne": ["Benzoyl Peroxide", "Salicylic Acid", "Topical Retinoids"],
    "Eczema": ["Moisturizers", "Topical Steroids", "Avoid Allergens"],
    "Psoriasis": ["Vitamin D Analogues", "Phototherapy", "Topical Steroids"],
    "Rosacea": ["Metronidazole Cream", "Gentle Skincare", "Avoid Triggers"],
    "Vitiligo": ["Topical Corticosteroids", "Phototherapy", "Skin Camouflage"],
    "Warts": ["Salicylic Acid", "Cryotherapy", "Laser Treatment"]
}

# ================= DESCRIPTIONS =================
descriptions = {
    "Acne": "Acne is a common skin condition that occurs when hair follicles become clogged with excess oil, dead skin cells, and bacteria.",
    "Eczema": "Eczema is a chronic inflammatory skin condition characterized by dry, itchy, and irritated skin.",
    "Psoriasis": "Psoriasis is an autoimmune skin disorder that causes rapid skin cell production, resulting in thick scaly patches.",
    "Rosacea": "Rosacea is a long-term skin condition causing facial redness and acne-like bumps.",
    "Vitiligo": "Vitiligo causes loss of skin pigment, resulting in white patches.",
    "Warts": "Warts are small skin growths caused by the human papillomavirus (HPV)."
}

# ================= SYMPTOMS =================
symptoms = {
    "Acne": ["Pimples and red bumps", "Blackheads and whiteheads", "Oily skin", "Painful cysts"],
    "Eczema": ["Dry and sensitive skin", "Severe itching", "Red patches", "Cracked skin"],
    "Psoriasis": ["Scaly red patches", "Dry skin", "Itching", "Nail changes"],
    "Rosacea": ["Facial redness", "Visible blood vessels", "Burning sensation"],
    "Vitiligo": ["White skin patches", "Loss of pigment", "Usually painless"],
    "Warts": ["Rough skin growths", "Skin-colored bumps", "Foot pain"]
}

# ================= IMAGE UPLOAD =================
uploaded_file = st.file_uploader(
    "📤 Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", width=350)

    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    disease_percentages = pred * 100
    idx = np.argmax(pred)

    predicted_class = class_names[idx]
    predicted_percent = disease_percentages[idx]

    # ================= RESULT =================
    st.markdown(f"""
    <div class="result-box">
        <h2>🩺 Detected Disease</h2>
        <div class="disease">{predicted_class}</div>
        <div class="percent">{predicted_percent:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # ================= DESCRIPTION =================
    st.markdown("<h2>📖 Disease Information</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="description-box">
        {descriptions[predicted_class]}
    </div>
    """, unsafe_allow_html=True)

    # ==========

    
    # ================= ✅ SYMPTOMS CARD (ADDED) =================
    st.markdown("<h2>🚨 Common Symptoms</h2>", unsafe_allow_html=True)

    symptom_html = "<ul style='text-align:left;'>"
    for s in symptoms[predicted_class]:
        symptom_html += f"<li>✔ {s}</li>"
    symptom_html += "</ul>"

    st.markdown(f"""
    <div class="description-box">
        {symptom_html}
    </div>
    """, unsafe_allow_html=True)

    # ================= BAR CHART =================
    st.markdown("<h2>📊 Disease Probability</h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(class_names, disease_percentages, color="#38bdf8")
    ax.set_ylabel("Percentage (%)")
    plt.xticks(rotation=35)
    st.pyplot(fig)

    # ================= TREATMENTS =================
    st.markdown("<h2>💊 Recommended Treatments</h2>", unsafe_allow_html=True)
    for t in recommendations[predicted_class]:
        st.markdown(f"""
        <div class="treatment-box">
            ✔ {t}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        f"<p style='text-align:center; font-size:13px; color:{SUBTEXT};'>"
        "⚠ Educational purpose only — consult a dermatologist."
        "</p>",
        unsafe_allow_html=True
    )
