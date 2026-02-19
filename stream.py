import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import zipfile
import tempfile
import os

# Page config
st.set_page_config(
    page_title="PawDetect AI",
    page_icon="🐾",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #1a1a2e 0%, #0a0a0f 60%) !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stMain"] > div { padding: 0 !important; }
.block-container {
    max-width: 680px !important;
    padding: 3rem 2rem 4rem !important;
    margin: 0 auto;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #f0a500;
    background: rgba(240,165,0,0.1);
    border: 1px solid rgba(240,165,0,0.25);
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2.8rem, 7vw, 4.2rem);
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: #f5f0e8;
    margin-bottom: 0.8rem;
}
.hero h1 span {
    background: linear-gradient(135deg, #f0a500 0%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    font-size: 1rem;
    color: #777;
    font-weight: 300;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.75;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    margin: 2rem 0;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(240,165,0,0.3) !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(240,165,0,0.55) !important;
    background: rgba(240,165,0,0.025) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span {
    color: #bbb !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(240,165,0,0.12) !important;
    color: #f0a500 !important;
    border: 1px solid rgba(240,165,0,0.35) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(240,165,0,0.22) !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    width: 100% !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    display: block;
}

/* ── Result Card ── */
.result-card {
    margin-top: 1.8rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2rem 2rem 1.6rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f0a500, #ff6b35);
}
.result-label {
    font-size: 0.66rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.9rem;
}
.result-animal {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    line-height: 1;
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem;
}
.result-animal.dog { color: #ff6b35; }
.result-animal.cat { color: #f0a500; }

.result-sub {
    font-size: 0.83rem;
    color: #555;
    margin-bottom: 1.6rem;
}

.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.45rem;
}
.conf-text { font-size: 0.76rem; color: #777; }
.conf-pct {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #f0a500;
}
.conf-bar-bg {
    height: 4px;
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #f0a500, #ff6b35);
}
.raw-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 1.3rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.38rem 0.75rem;
    font-size: 0.73rem;
    color: #444;
    font-family: monospace;
}
.raw-chip span { color: #777; }

.footer {
    text-align: center;
    padding-top: 3.5rem;
    font-size: 0.7rem;
    color: #2a2a2a;
    letter-spacing: 0.07em;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🐾 &nbsp; Computer Vision · Deep Learning</div>
    <h1>Paw<span>Detect</span></h1>
    <p>Drop a photo and our neural network will instantly tell you — cat or dog?</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    model_path = "cat_dog_model_new.keras"
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except TypeError as e:
        if "batch_shape" not in str(e):
            raise e
        with zipfile.ZipFile(model_path, "r") as z:
            files = {name: z.read(name) for name in z.namelist()}
        config_str = files["config.json"].decode("utf-8").replace('"batch_shape"', '"shape"')
        files["config.json"] = config_str.encode("utf-8")
        patched_path = tempfile.mktemp(suffix=".keras")
        with zipfile.ZipFile(patched_path, "w") as z:
            for name, data in files.items():
                z.writestr(name, data)
        model = tf.keras.models.load_model(patched_path, compile=False)
        os.remove(patched_path)
        return model

model = load_my_model()
IMG_SIZE = 150


# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ── Prediction ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    st.image(image, use_column_width=True)

    img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analysing…"):
        prediction = model.predict(img_array, verbose=0)

    raw = float(prediction[0][0])

    if raw > 0.5:
        animal, emoji, css_class, conf = "Dog", "🐶", "dog", raw
    else:
        animal, emoji, css_class, conf = "Cat", "🐱", "cat", 1 - raw

    pct = int(conf * 100)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Prediction Result</div>
        <div class="result-animal {css_class}">{emoji} {animal}</div>
        <div class="result-sub">Identified with {pct}% confidence</div>

        <div class="conf-row">
            <span class="conf-text">Confidence Score</span>
            <span class="conf-pct">{pct}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct}%"></div>
        </div>

        <div class="raw-chip">raw output &nbsp;·&nbsp; <span>{raw:.6f}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PawDetect AI &nbsp;·&nbsp; Powered by TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)