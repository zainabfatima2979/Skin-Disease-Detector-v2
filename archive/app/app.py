import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import InputLayer
from pymongo import MongoClient
from datetime import datetime
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Skin Disease Detector",
    page_icon="🩺",
    layout="wide"
)

# ================= THEME SELECT (MOON / SUN STYLE) =================
if "theme" not in st.session_state:
    st.session_state.theme = "Dark Mode"

theme_label = "Dark Mode" if st.session_state.theme == "Dark Mode" else "Light Mode"

st.sidebar.markdown(
    """
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🩺</div>
        <div class="sidebar-logo-text">
            <span class="sidebar-logo-title">DermaScan AI</span>
            <span class="sidebar-logo-sub">Skin Diagnosis Assistant</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<h3 style='text-align:center;'>Navigation</h3>",
    unsafe_allow_html=True
)

nav_page = st.sidebar.radio(
    "",
    ["Detector", "Records"]
)

st.sidebar.markdown(
    "<h3 style='text-align:center;'>Appearance</h3>",
    unsafe_allow_html=True
)

theme = st.sidebar.radio(
    "",
    ["Dark Mode", "Light Mode"],
    index=0 if st.session_state.theme == "Dark Mode" else 1,
    key="appearance_radio"
)

st.session_state.theme = theme

st.sidebar.markdown("<hr class='sidebar-divider'/>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div class="sidebar-info">
        <p class="sidebar-info-title">How it works</p>
        <ol class="sidebar-info-list">
            <li>Upload a clear photo of the affected skin area</li>
            <li>The AI model analyzes the image</li>
            <li>Get the predicted condition, symptoms & care tips</li>
        </ol>
        <p class="sidebar-info-title">Detects</p>
        <div class="sidebar-tags">
            <span class="tag">Acne</span>
            <span class="tag">Eczema</span>
            <span class="tag">Psoriasis</span>
            <span class="tag">Rosacea</span>
            <span class="tag">Vitiligo</span>
            <span class="tag">Warts</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


if theme == "Dark Mode":
    BG = "#0b0f19"
    BG_GRADIENT = "radial-gradient(circle at 15% 10%, rgba(56,189,248,0.16), transparent 40%), radial-gradient(circle at 85% 20%, rgba(168,85,247,0.14), transparent 45%), radial-gradient(circle at 50% 100%, rgba(34,211,238,0.10), transparent 50%), #0b0f19"
    CARD = "linear-gradient(145deg, rgba(15,23,42,0.85), rgba(30,41,59,0.85))"
    CARD_BORDER = "rgba(148,163,184,0.15)"
    TEXT = "#e5e7eb"
    SUBTEXT = "#94a3b8"
    TITLE = "#60a5fa"
    ACCENT = "#22d3ee"
    ACCENT2 = "#a855f7"
    GLOW = "rgba(56,189,248,0.35)"
    GAUGE_TRACK = "rgba(148,163,184,0.15)"
else:
    BG = "#f4f7fb"
    BG_GRADIENT = "radial-gradient(circle at 15% 10%, rgba(37,99,235,0.08), transparent 40%), radial-gradient(circle at 85% 20%, rgba(168,85,247,0.07), transparent 45%), #f4f7fb"
    CARD = "linear-gradient(145deg, #ffffff, #f8fafc)"
    CARD_BORDER = "rgba(15,23,42,0.08)"
    TEXT = "#0f172a"
    SUBTEXT = "#475569"
    TITLE = "#1e3a8a"
    ACCENT = "#0ea5e9"
    ACCENT2 = "#7c3aed"
    GLOW = "rgba(37,99,235,0.18)"
    GAUGE_TRACK = "rgba(15,23,42,0.08)"

# ================= FULL PAGE CSS =================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
@property --p {{
  syntax: '<number>';
  inherits: true;
  initial-value: 0;
}}

* {{
    font-family: 'Inter', 'Poppins', sans-serif;
}}

html, body, .stApp {{
    background: {BG_GRADIENT} !important;
    color: {TEXT} !important;
    background-attachment: fixed !important;
}}

header, section.main > div {{
    background: transparent !important;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

::-webkit-scrollbar {{ width: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {ACCENT}; border-radius: 10px; opacity: 0.6; }}

label {{
    color: {TEXT} !important;
    font-weight: 600;
}}

/* ============ STREAMLIT BUTTON OVERRIDE ============ */
div.stButton > button {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2}) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}
div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 18px {GLOW};
    color: #ffffff !important;
}}
div.stButton > button p, div.stButton > button span {{
    color: #ffffff !important;
}}

/* ============ ANIMATIONS ============ */
@keyframes fadeSlideUp {{
    from {{ opacity: 0; transform: translateY(24px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
}}
@keyframes floatUpDown {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-8px); }}
}}
@keyframes pulseGlow {{
    0%, 100% {{ box-shadow: 0 0 18px {GLOW}; }}
    50% {{ box-shadow: 0 0 34px {GLOW}; }}
}}
@keyframes growBar {{
    from {{ transform: scaleX(0); }}
    to {{ transform: scaleX(1); }}
}}
@keyframes popIn {{
    0% {{ opacity: 0; transform: scale(0.85); }}
    100% {{ opacity: 1; transform: scale(1); }}
}}

/* ============ HERO ============ */
.hero-wrap {{
    text-align: center;
    padding: 18px 0 8px 0;
    animation: fadeSlideUp 0.8s ease-out both;
}}
.hero-badge {{
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    background: linear-gradient(90deg, {ACCENT}22, {ACCENT2}22);
    border: 1px solid {ACCENT}55;
    color: {ACCENT};
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
    animation: floatUpDown 3.5s ease-in-out infinite;
}}
h1.app-title {{
    font-family: 'Poppins', sans-serif !important;
    background: linear-gradient(90deg, {TITLE}, {ACCENT2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}}
p.app-subtitle {{
    text-align: center;
    color: {SUBTEXT};
    font-size: 17px;
    font-weight: 500;
    margin-bottom: 4px;
}}

h2 {{
    font-family: 'Poppins', sans-serif !important;
    color: {TITLE} !important;
    text-align: center;
    font-weight: 700 !important;
    margin-top: 8px !important;
}}

/* ============ FILE UPLOADER ============ */
div[data-testid="stFileUploader"] {{
    background: {CARD} !important;
    border: 2px dashed {ACCENT}66 !important;
    border-radius: 22px !important;
    padding: 22px !important;
    box-shadow: 0 8px 30px {GLOW};
    transition: all 0.35s ease;
    animation: fadeSlideUp 0.9s ease-out both;
}}
div[data-testid="stFileUploader"]:hover {{
    border-color: {ACCENT} !important;
    transform: translateY(-3px);
    box-shadow: 0 12px 40px {GLOW};
}}
div[data-testid="stFileUploaderDropzone"] {{
    background: transparent !important;
}}
div[data-testid="stFileUploaderDropzoneInstructions"] {{
    color: {TEXT} !important;
}}
div[data-testid="stFileUploaderDropzoneInstructions"] span,
div[data-testid="stFileUploaderDropzoneInstructions"] small,
div[data-testid="stFileUploaderDropzoneInstructions"] div {{
    color: {TEXT} !important;
    opacity: 1 !important;
}}
div[data-testid="stFileUploaderDropzoneInstructions"] small {{
    color: {SUBTEXT} !important;
}}
div[data-testid="stFileUploaderDropzoneInstructions"] svg {{
    fill: {ACCENT} !important;
}}
div[data-testid="stFileUploader"] section {{
    background: transparent !important;
}}
div[data-testid="stFileUploader"] button {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2}) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}
div[data-testid="stFileUploader"] button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 18px {GLOW};
}}
div[data-testid="stFileUploader"] button p {{
    color: #ffffff !important;
}}

/* ============ UPLOADED IMAGE ============ */
div[data-testid="stImage"] {{
    animation: popIn 0.6s ease-out both;
}}
div[data-testid="stImage"] img {{
    border-radius: 20px !important;
    box-shadow: 0 10px 34px {GLOW};
    border: 1px solid {CARD_BORDER};
}}

/* ============ GENERIC CARD ============ */
.section-card {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 22px;
    padding: 26px 30px;
    margin: 18px auto;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    animation: fadeSlideUp 0.7s ease-out both;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}
.section-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 16px 40px {GLOW};
}}

/* ============ RESULT / GAUGE ============ */
.result-box {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    padding: 36px 24px;
    border-radius: 26px;
    margin-top: 24px;
    box-shadow: 0 12px 40px {GLOW};
    text-align: center;
    animation: fadeSlideUp 0.8s ease-out both, pulseGlow 4s ease-in-out infinite 1s;
}}
.result-eyebrow {{
    font-family: 'Poppins', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {ACCENT};
    margin: 0 0 6px 0;
}}
.disease {{
    font-family: 'Poppins', sans-serif;
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 4px;
}}

.gauge-outer {{
    --p: 0;
    width: 176px;
    height: 176px;
    border-radius: 50%;
    margin: 18px auto 8px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    background: conic-gradient(from -90deg, {ACCENT} calc(var(--p) * 3.6deg), {GAUGE_TRACK} 0deg);
    animation: fillGauge 1.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    box-shadow: 0 0 30px {GLOW};
}}
@keyframes fillGauge {{
    to {{ --p: var(--target); }}
}}
.gauge-inner {{
    width: 138px;
    height: 138px;
    border-radius: 50%;
    background: {BG};
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 0 12px rgba(0,0,0,0.15);
}}
.gauge-value {{
    font-size: 30px;
    font-weight: 800;
    font-family: 'Poppins', sans-serif;
    color: {TEXT};
}}
.gauge-label {{
    font-size: 12px;
    color: {SUBTEXT};
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}}

/* ============ CONFIDENCE BAR ============ */
.confidence-track {{
    width: 80%;
    max-width: 420px;
    height: 10px;
    background: {GAUGE_TRACK};
    border-radius: 999px;
    margin: 14px auto 0 auto;
    overflow: hidden;
}}
.confidence-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    transform-origin: left;
    animation: growBar 1.4s cubic-bezier(0.22, 1, 0.36, 1) both;
}}

/* ============ DESCRIPTION ============ */
.description-box {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    padding: 24px 30px;
    margin: 16px auto;
    max-width: 780px;
    border-radius: 20px;
    color: {TEXT};
    text-align: center;
    line-height: 1.7;
    box-shadow: 0 8px 26px rgba(0,0,0,0.10);
    font-size: 16px;
    animation: fadeSlideUp 0.7s ease-out both;
}}

/* ============ SYMPTOMS GRID ============ */
.symptom-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 14px;
    max-width: 900px;
    margin: 16px auto;
}}
.symptom-chip {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-left: 4px solid {ACCENT};
    border-radius: 14px;
    padding: 14px 16px;
    text-align: left;
    font-size: 14.5px;
    color: {TEXT};
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeSlideUp 0.6s ease-out both;
}}
.symptom-chip:hover {{
    transform: translateY(-3px) scale(1.015);
    box-shadow: 0 10px 26px {GLOW};
    border-left-color: {ACCENT2};
}}
.symptom-check {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    margin-right: 10px;
    border-radius: 5px;
    background: {ACCENT}22;
    color: {ACCENT};
    font-size: 11px;
    font-weight: 700;
}}

/* ============ TREATMENT GRID ============ */
.treatment-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
    max-width: 1000px;
    margin: 16px auto;
}}
.treatment-box {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    padding: 18px 18px;
    border-radius: 16px;
    color: {TEXT};
    text-align: left;
    box-shadow: 0 8px 22px rgba(34,197,94,0.12);
    display: flex;
    gap: 12px;
    align-items: flex-start;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeSlideUp 0.65s ease-out both;
    font-size: 14.5px;
    line-height: 1.5;
}}
.treatment-box:hover {{
    transform: translateY(-4px);
    box-shadow: 0 14px 32px rgba(34,197,94,0.25);
}}
.treatment-num {{
    flex-shrink: 0;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, {ACCENT}, #22c55e);
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 13px;
}}

/* ============ CHART CARD ============ */
.chart-card {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 22px;
    padding: 12px 18px 4px 18px;
    max-width: 780px;
    margin: 16px auto;
    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
    animation: fadeSlideUp 0.7s ease-out both;
}}

/* ============ FOOTER DISCLAIMER ============ */
.disclaimer-box {{
    max-width: 620px;
    margin: 30px auto 12px auto;
    padding: 14px 20px;
    border-radius: 14px;
    text-align: center;
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-left: 4px solid #f59e0b;
    color: {SUBTEXT};
    font-size: 13px;
    animation: fadeIn 1s ease-out both;
}}

/* ============ SIDEBAR ============ */
section[data-testid="stSidebar"] {{
    background: {BG} !important;
    border-right: 1px solid {CARD_BORDER};
}}
section[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
}}
section[data-testid="stSidebar"] label {{
    color: {TEXT} !important;
    font-weight: 600;
}}
.sidebar-logo {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 4px 16px 4px;
    animation: fadeSlideUp 0.6s ease-out both;
}}
.sidebar-logo-icon {{
    font-size: 30px;
    animation: floatUpDown 3s ease-in-out infinite;
}}
.sidebar-logo-text {{
    display: flex;
    flex-direction: column;
    line-height: 1.15;
}}
.sidebar-logo-title {{
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 17px;
    color: {TITLE} !important;
}}
.sidebar-logo-sub {{
    font-size: 11.5px;
    color: {SUBTEXT} !important;
}}
.sidebar-divider {{
    border: none;
    border-top: 1px solid {CARD_BORDER};
    margin: 14px 0;
}}
.sidebar-info {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 16px 16px 18px 16px;
    animation: fadeSlideUp 0.7s ease-out both;
}}
.sidebar-info-title {{
    font-weight: 700 !important;
    font-size: 13.5px;
    margin-bottom: 6px !important;
    color: {TITLE} !important;
}}
.sidebar-info-list {{
    font-size: 12.5px;
    padding-left: 18px;
    color: {SUBTEXT} !important;
    margin-bottom: 14px;
}}
.sidebar-info-list li {{
    margin-bottom: 4px;
}}
.sidebar-tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}}
.tag {{
    background: {ACCENT}1f;
    border: 1px solid {ACCENT}55;
    color: {ACCENT} !important;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 999px;
}}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
class FixedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)
        
MODEL_PATH = r"C:\Users\Administrator\Desktop\Skin-Disease-Detector\archive\models\skin_detector_best_7cls.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects={'InputLayer': FixedInputLayer})

class_names = ["Acne", "Eczema", "Psoriasis", "Rosacea", "Vitiligo", "Warts"]
IMG_SIZE = 224

# ================= TREATMENTS =================
recommendations = {
    "Acne": [
        "Benzoyl Peroxide: Apply once or twice daily on affected areas to kill acne-causing bacteria.",
        "Salicylic Acid: Use once daily to unclog pores and reduce redness.",
        "Topical Retinoids: Apply once at night before sleeping to prevent new pimples.",
        "Gentle Face Wash: Wash face twice daily (morning and night).",
        "Oil-Free Moisturizer: Apply twice daily after washing face."
    ],
    "Eczema": [
        "Moisturizers: Apply 2–3 times daily to keep skin hydrated.",
        "Topical Steroids: Apply once or twice daily during flare-ups as prescribed.",
        "Avoid Allergens: Follow daily to prevent irritation.",
        "Mild Soap: Use once daily during bathing.",
        "Lukewarm Bath: Take once daily for 5–10 minutes only."
    ],
    "Psoriasis": [
        "Vitamin D Analogues: Apply once daily on affected skin.",
        "Topical Steroids: Apply once or twice daily depending on severity.",
        "Phototherapy: 2–3 sessions per week under medical supervision.",
        "Moisturizing Creams: Apply at least twice daily.",
        "Stress Control Techniques: Practice daily (relaxation, exercise)."
    ],
    "Rosacea": [
        "Metronidazole Cream: Apply once or twice daily on facial redness.",
        "Gentle Skincare Products: Use daily, morning and night.",
        "Sunscreen: Apply once daily before going outside.",
        "Avoid Triggers: Follow daily (avoid spicy food, heat, caffeine).",
        "Wash Face with Lukewarm Water: Twice daily."
    ],
    "Vitiligo": [
        "Topical Corticosteroids: Apply once daily on white patches.",
        "Phototherapy: 2–3 times per week under doctor guidance.",
        "Skin Camouflage Makeup: Use daily as needed.",
        "Sunscreen: Apply daily before sun exposure.",
        "Psychological Support Activities: Practice daily for mental well-being."
    ],
    "Warts": [
        "Salicylic Acid Treatment: Apply once daily directly on wart.",
        "Cryotherapy: Performed every 2–3 weeks by a doctor.",
        "Laser Treatment: Usually one session if required.",
        "Avoid Touching Warts: Follow daily to prevent spreading.",
        "Keep Area Clean and Dry: Clean once or twice daily."
    ],
    "Scabies": [
        "Permethrin Cream (5%): Apply once at night on the whole body from neck down, wash off after 8–14 hours.",
        "Ivermectin (Oral): As prescribed by a doctor, usually a single dose repeated after 1–2 weeks.",
        "Antihistamines: Take as needed to reduce itching, especially at night.",
        "Wash Clothing & Bedding: Wash all clothes, towels, and bedding in hot water daily during treatment.",
        "Treat Close Contacts: Family members/close contacts should be treated simultaneously to prevent reinfection."
    ]
}

# ================= DESCRIPTIONS =================
descriptions = {
    "Acne": "Acne is a very common skin condition that mainly affects teenagers and young adults, but it can occur at any age. It develops when hair follicles become blocked by excess oil (sebum), dead skin cells, and bacteria. This blockage leads to pimples, blackheads, whiteheads, and sometimes painful cysts. Hormonal changes, stress, oily skin, and poor skincare can make acne worse.",
    "Eczema": "Eczema, also known as atopic dermatitis, is a chronic skin condition that causes inflammation, dryness, and intense itching. The skin becomes red, cracked, and sensitive, making it more prone to infections. Eczema often appears in childhood but can continue into adulthood. It is commonly triggered by allergens, harsh soaps, weather changes, or stress.",
    "Psoriasis": "Psoriasis is a long-term autoimmune skin disorder in which the immune system causes skin cells to grow much faster than normal. This rapid growth results in thick, red patches covered with silvery scales. Psoriasis can cause itching, burning, and discomfort. Stress, infections, cold weather, and certain medications can trigger flare-ups.",
    "Rosacea": "Rosacea is a chronic skin condition that mainly affects the face, causing redness, visible blood vessels, and acne-like bumps. It usually appears in adults and may worsen over time if left untreated. Common triggers include sun exposure, spicy foods, hot drinks, stress, and extreme temperatures. Rosacea does not have a permanent cure, but proper care can control symptoms.",
    "Scabies": "Scabies is a contagious skin infestation caused by the microscopic mite Sarcoptes scabiei, which burrows into the top layer of skin to lay eggs. It causes intense itching, especially at night, and a pimple-like rash. Scabies spreads quickly through prolonged skin-to-skin contact or shared bedding/clothing, making household and close-contact treatment essential.",
    "Vitiligo": "Vitiligo is a skin disorder in which the skin loses its natural color due to damage or loss of pigment-producing cells called melanocytes. This results in white patches on different parts of the body. Vitiligo is not contagious and can affect people of any age. Although it does not cause physical pain, it can impact emotional and psychological well-being.",
    "Warts": "Warts are small, rough, and usually painless skin growths caused by infection with the human papillomavirus (HPV). They can appear on hands, feet, face, or other parts of the body. Warts spread through direct contact or shared surfaces. While many warts disappear on their own, some require medical treatment to remove."
}

# ================= SYMPTOMS =================
symptoms = {
    "Acne": ["Pimples and red inflamed bumps", "Blackheads and whiteheads", "Oily or greasy skin", "Painful cysts under the skin", "Swollen and tender areas", "Dark spots or acne scars after healing", "Breakouts on face, chest, and back"],
    "Eczema": ["Dry, rough, and sensitive skin", "Severe itching, especially at night", "Red or brownish patches", "Cracked or scaly skin", "Skin thickening due to scratching", "Oozing or crusting in severe cases", "Increased skin sensitivity"],
    "Psoriasis": ["Thick scaly red patches", "Dry and cracked skin that may bleed", "Persistent itching or burning", "Silvery-white scales on skin", "Joint pain in severe cases", "Nail changes such as pitting or discoloration", "Skin soreness and irritation"],
    "Rosacea": ["Persistent facial redness", "Visible small blood vessels on the face", "Burning or stinging sensation", "Acne-like bumps and pimples", "Facial swelling", "Dry and sensitive facial skin", "Eye irritation in some cases"],
    "Scabies": ["Intense itching, especially at night", "Pimple-like rash or small red bumps", "Thin, irregular burrow tracks on the skin", "Common between fingers, wrists, and skin folds", "Sores caused by scratching", "Thickened, crusted skin in severe cases", "Rash that spreads to other family members"],
    "Vitiligo": ["White patches on skin", "Loss of natural skin color", "Premature whitening of hair", "Patches around mouth, eyes, or joints", "Symmetrical patch appearance", "No itching or pain in most cases", "Increased sensitivity to sunlight"],
    "Warts": ["Rough and raised skin growths", "Skin-colored or dark bumps", "Pain or discomfort when pressed", "Clusters of small warts", "Bleeding if scratched", "Common on hands and feet", "Hard or grainy surface texture"]
}

# ================= IMAGE VALIDITY FILTER =================
def get_skin_percentage(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 25, 60], dtype=np.uint8)
    upper1 = np.array([22, 170, 245], dtype=np.uint8)
    mask1 = cv2.inRange(img_hsv, lower1, upper1)

    lower2 = np.array([165, 25, 60], dtype=np.uint8)
    upper2 = np.array([180, 170, 245], dtype=np.uint8)
    mask2 = cv2.inRange(img_hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    skin_pixels = cv2.countNonZero(mask)
    return (skin_pixels / total_pixels) * 100

# ================= REFINED CONTOUR HIGHLIGHT FUNCTION =================
def detect_and_draw_lesions(img_rgb):
    output_img = img_rgb.copy()
    img_hsv = cv2.cvtColor(output_img, cv2.COLOR_RGB2HSV)
    
    lower_red1 = np.array([0, 25, 50], dtype=np.uint8)
    upper_red1 = np.array([15, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)

    lower_red2 = np.array([165, 25, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    lab = cv2.cvtColor(output_img, cv2.COLOR_RGB2LAB)
    _, a_channel, _ = cv2.split(lab)
    _, a_thresh = cv2.threshold(a_channel, 138, 255, cv2.THRESH_BINARY)
    
    combined = cv2.bitwise_and(red_mask, a_thresh)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150:
            (x, y, w, h) = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 1
            if 0.2 < aspect_ratio < 5.0:
                cv2.drawContours(output_img, [cnt], -1, (0, 255, 255), 2)
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 165, 255), 2)

    return output_img

# ================= SAVE TO MONGODB FUNCTION =================
def save_prediction_to_mongodb(image_name, disease, confidence, image_bytes=None):
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["skin_disease_db"]
        collection = db["predictions_record"]
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        record_data = {
            "date_time": current_time,
            "image_name": image_name,
            "detected_disease": disease,
            "confidence_percentage": float(confidence),
            "image_bytes": image_bytes  # Storing image bytes for record details view
        }
        
        collection.insert_one(record_data)
    except Exception as e:
        print(f"MongoDB Error: {e}")

SKIN_PERCENT_THRESHOLD = 20
CONFIDENCE_THRESHOLD = 30

if nav_page == "Detector":
    # ================= TITLE =================
    st.markdown(
        """
        <div class="hero-wrap">
            <span class="hero-badge">AI-Powered Dermatology Assistant</span>
            <h1 class="app-title">Skin Disease Detector</h1>
            <p class="app-subtitle">Upload a skin image and let AI predict the condition instantly</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ================= IMAGE UPLOAD =================
    uploaded_file = st.file_uploader(
        "Upload Skin Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        raw_bytes = uploaded_file.read()
        file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        skin_percent = get_skin_percentage(img_rgb)

        if skin_percent < SKIN_PERCENT_THRESHOLD:
            # Save record for non-skin / invalid image rejections
            save_prediction_to_mongodb(uploaded_file.name, "No Skin Condition Recognized (Invalid)", 0.0, raw_bytes)

            col_a, col_b, col_c = st.columns([1, 1.3, 1])
            with col_b:
                st.image(img_rgb, caption="Uploaded Image", width=350)
            st.markdown(
                """
                <div class="result-box">
                    <p class="result-eyebrow">Not Detected</p>
                    <div class="disease" style="font-size:26px;">No Skin Condition Recognized</div>
                    <p style="color:#94a3b8; margin-top:14px; font-size:14.5px;">
                        This image doesn't appear to show human skin. Please upload a clear,
                        close-up photo of the affected skin area.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.stop()

        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_array = img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing image and marking lesions..."):
            pred = model.predict(img_array)[0][:len(class_names)]
            marked_img = detect_and_draw_lesions(img_rgb)

        disease_percentages = pred * 100
        idx = np.argmax(pred)

        predicted_class = class_names[idx]
        predicted_percent = disease_percentages[idx]

        # Show images side by side (Original vs Marked with Contours)
        col_1, col_2 = st.columns(2)
        with col_1:
            st.image(img_rgb, caption="Original Uploaded Image", use_container_width=True)
        with col_2:
            st.image(marked_img, caption="AI Detected Lesion / Affected Area (Marked)", use_container_width=True)

        if predicted_percent < CONFIDENCE_THRESHOLD:
            # Save record for low-confidence rejections
            save_prediction_to_mongodb(uploaded_file.name, f"Uncertain ({predicted_class})", predicted_percent, raw_bytes)

            st.markdown(
                f"""
                <div class="result-box">
                    <p class="result-eyebrow">Not Detected</p>
                    <div class="disease" style="font-size:26px;">Unable to Confidently Classify</div>
                    <p style="color:#94a3b8; margin-top:14px; font-size:14.5px;">
                        The model isn't confident enough about this image (highest match was only
                        {predicted_percent:.1f}%). Please try a clearer, well-lit, close-up photo
                        of the affected skin area.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.stop()

        # Save record to MongoDB automatically when successfully classified including raw bytes
        save_prediction_to_mongodb(uploaded_file.name, predicted_class, predicted_percent, raw_bytes)

        # ================= RESULT =================
        st.markdown(f"""
        <div class="result-box">
            <p class="result-eyebrow">Detected Condition</p>
            <div class="disease">{predicted_class}</div>
            <div class="gauge-outer" style="--target:{predicted_percent:.2f};">
                <div class="gauge-inner">
                    <div class="gauge-value">{predicted_percent:.1f}%</div>
                    <div class="gauge-label">Confidence</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ================= DESCRIPTION =================
        st.markdown("<h2>Disease Information</h2>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="description-box">
            {descriptions[predicted_class]}
        </div>
        """, unsafe_allow_html=True)

        # ================= SYMPTOMS CARD =================
        st.markdown("<h2>Common Symptoms</h2>", unsafe_allow_html=True)
        symptom_cards = "".join(
            f"<div class='symptom-chip' style='animation-delay:{i * 0.06}s;'><span class='symptom-check'>&#10003;</span>{s}</div>"
            for i, s in enumerate(symptoms[predicted_class])
        )
        st.markdown(f"<div class='symptom-grid'>{symptom_cards}</div>", unsafe_allow_html=True)

        # ================= BAR CHART =================
        st.markdown("<h2>Disease Probability</h2>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)

        plt.rcParams["font.family"] = "sans-serif"
        fig, ax = plt.subplots(figsize=(6.8, 4.1))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        bar_colors = ["#a855f7" if name != predicted_class else "#22d3ee" for name in class_names]
        bars = ax.bar(class_names, disease_percentages, color=bar_colors, width=0.55, edgecolor="none", zorder=3)

        for bar, val in zip(bars, disease_percentages):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=9, color=TEXT, fontweight="bold")

        ax.set_ylabel("Percentage (%)", color=SUBTEXT, fontsize=10)
        ax.set_ylim(0, max(disease_percentages) + 15)
        ax.tick_params(colors=SUBTEXT, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SUBTEXT)
        ax.spines["bottom"].set_color(SUBTEXT)
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        plt.xticks(rotation=25, color=TEXT)

        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        # ================= TREATMENTS =================
        st.markdown("<h2>Recommended Treatments</h2>", unsafe_allow_html=True)
        treatment_cards = "".join(
            f"<div class='treatment-box' style='animation-delay:{i * 0.07}s;'>"
            f"<div class='treatment-num'>{i + 1}</div><div>{t}</div></div>"
            for i, t in enumerate(recommendations[predicted_class])
        )
        st.markdown(f"<div class='treatment-grid'>{treatment_cards}</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="disclaimer-box">
                <strong>Disclaimer:</strong> This tool is for educational purposes only and is not a substitute for
                professional medical advice. Please consult a dermatologist for diagnosis and treatment.
            </div>
            """,
            unsafe_allow_html=True
        )

elif nav_page == "Records":
    # Using session state to handle single-record view mode so other records hide when viewed
    if "selected_record" not in st.session_state:
        st.session_state.selected_record = None

    if st.session_state.selected_record is not None:
        # ================= DEDICATED RECORD DETAIL PAGE =================
        if st.button("← Back to Records List"):
            st.session_state.selected_record = None
            st.rerun()

        rec = st.session_state.selected_record
        disease = rec.get('detected_disease', 'N/A')

        st.markdown(
            f"""
            <div class="hero-wrap">
                <span class="hero-badge">Record Details</span>
                <h1 class="app-title">{rec.get('image_name', 'Record')}</h1>
                <p class="app-subtitle">Detailed diagnosis report saved on {rec.get('date_time', 'N/A')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display stored image if available
        img_bytes = rec.get('image_bytes')
        if img_bytes:
            nparr = np.frombuffer(img_bytes, np.uint8)
            detail_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            detail_img_rgb = cv2.cvtColor(detail_img, cv2.COLOR_BGR2RGB)
            
            col_1, col_2, col_3 = st.columns([1, 1.5, 1])
            with col_2:
                st.image(detail_img_rgb, caption=rec.get('image_name', 'Uploaded Image'), use_container_width=True)

        # Result Summary Box
        conf = rec.get('confidence_percentage', 0.0)
        st.markdown(f"""
        <div class="result-box">
            <p class="result-eyebrow">Diagnosed Condition</p>
            <div class="disease">{disease}</div>
            <p style="color:{SUBTEXT}; margin-top:10px; font-size:15px;">Confidence Score: <b>{conf:.1f}%</b></p>
            <p style="color:{SUBTEXT}; font-size:13.5px;">Recorded At: {rec.get('date_time', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Show Description if available in dict
        if disease in descriptions:
            st.markdown("<h2>Disease Information</h2>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="description-box">
                {descriptions[disease]}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<h2>Common Symptoms</h2>", unsafe_allow_html=True)
            symptom_cards = "".join(
                f"<div class='symptom-chip' style='animation-delay:{i * 0.06}s;'><span class='symptom-check'>&#10003;</span>{s}</div>"
                for i, s in enumerate(symptoms[disease])
            )
            st.markdown(f"<div class='symptom-grid'>{symptom_cards}</div>", unsafe_allow_html=True)

            st.markdown("<h2>Recommended Treatments</h2>", unsafe_allow_html=True)
            treatment_cards = "".join(
                f"<div class='treatment-box' style='animation-delay:{i * 0.07}s;'>"
                f"<div class='treatment-num'>{i + 1}</div><div>{t}</div></div>"
                for i, t in enumerate(recommendations[disease])
            )
            st.markdown(f"<div class='treatment-grid'>{treatment_cards}</div>", unsafe_allow_html=True)

    else:
        # ================= RECORDS LIST PAGE =================
        st.markdown(
            """
            <div class="hero-wrap">
                <span class="hero-badge">Database Records</span>
                <h1 class="app-title">Diagnosis History</h1>
                <p class="app-subtitle">View all saved records stored in MongoDB</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:
            client = MongoClient("mongodb://localhost:27017/")
            db = client["skin_disease_db"]
            collection = db["predictions_record"]
            records = list(collection.find().sort("date_time", -1))
            
            if not records:
                st.info("No records found in the database.")
            else:
                for idx, rec in enumerate(records):
                    with st.container():
                        st.markdown(
                            f"""
                            <div class="section-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong style="font-size:16px; color:{TITLE};">{rec.get('image_name', 'Unknown Image')}</strong>
                                        <p style="margin: 4px 0 0 0; color:{SUBTEXT}; font-size:13px;">📅 {rec.get('date_time', 'N/A')} &nbsp;|&nbsp; 🦠 <b>{rec.get('detected_disease', 'N/A')}</b> ({rec.get('confidence_percentage', 0.0):.1f}%)</p>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button("View Details", key=f"view_btn_{idx}"):
                            st.session_state.selected_record = rec
                            st.rerun()
        except Exception as e:
            st.error(f"Failed to connect to MongoDB: {e}")