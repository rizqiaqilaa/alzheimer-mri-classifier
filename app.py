import streamlit as st
import numpy as np
import cv2
import joblib
import pywt
from PIL import Image
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.draw import polygon
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import io


# PAGE CONFIG
st.set_page_config(
    page_title="Alzheimer MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

# CONSTANTS
CLASS_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
CLASS_LABELS = {
    "NonDemented":      ("Non Demented",       "🟢", "#28a745"),
    "VeryMildDemented": ("Very Mild Demented",  "🟡", "#ffc107"),
    "MildDemented":     ("Mild Demented",       "🟠", "#fd7e14"),
    "ModerateDemented": ("Moderate Demented",   "🔴", "#dc3545"),
}
SEVERITY_ORDER = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

# LOAD MODELS (cached)
@st.cache_resource
def load_models():
    svm    = joblib.load("results/best_svm.pkl")
    rf     = joblib.load("results/best_rf.pkl")
    scaler = joblib.load("results/scaler_fusion.pkl")
    return svm, rf, scaler

# PREPROCESSING FUNCTIONS 
def preprocess_image_array(img_array, size=128):
    """Preprocess numpy array (RGB) the same way as notebook."""
    img_resized = cv2.resize(img_array, (size, size))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(blur)

    # Active contour segmentation
    img_float = cl / 255.0
    img_smooth = gaussian(img_float, sigma=3)

    h, w = img_float.shape
    cy, cx = h // 2, w // 2
    r_init = min(h, w) // 2 - 5
    s = np.linspace(0, 2 * np.pi, 200)
    init_contour = np.array([
        cy + r_init * np.cos(s),
        cx + r_init * np.sin(s)
    ]).T

    snake = active_contour(
        img_smooth, init_contour,
        alpha=0.015, beta=10, gamma=0.001,
        w_line=0, w_edge=1, max_num_iter=100
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    rr, cc = polygon(snake[:, 0].astype(int), snake[:, 1].astype(int), (h, w))
    rr = np.clip(rr, 0, h - 1)
    cc = np.clip(cc, 0, w - 1)
    mask[rr, cc] = 255

    masked = cv2.bitwise_and(cl, cl, mask=mask)
    norm = masked / 255.0
    return norm, mask, cl


def build_gabor_bank(orientations=12, scales=7, ksize=21, sigma=4.0, gamma=0.5, psi=0):
    filters = []
    wavelengths = np.logspace(np.log10(3), np.log10(40), scales)
    for theta_idx in range(orientations):
        theta = theta_idx * np.pi / orientations
        for lam in wavelengths:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma=sigma, theta=theta,
                lambd=lam, gamma=gamma, psi=psi, ktype=cv2.CV_64F
            )
            kernel /= (kernel.sum() + 1e-10)
            filters.append((kernel, theta, lam))
    return filters

GABOR_BANK = build_gabor_bank()

def extract_gabor_features(image):
    img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    features = []
    for kernel, _, _ in GABOR_BANK:
        response = cv2.filter2D(img_uint8, cv2.CV_64F, kernel)
        mag = np.abs(response).ravel()
        mean    = np.mean(mag)
        var     = np.var(mag)
        energy  = np.sum(mag ** 2)
        prob    = (mag + 1e-10) / (mag.sum() + 1e-10)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        features.extend([mean, var, energy, entropy])
    return np.array(features, dtype=np.float32)

def extract_lbp_features(image, radii=[1, 2, 3, 4, 5]):
    img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    features = []
    for r in radii:
        n_pts = 8 * r
        lbp_map = local_binary_pattern(img_uint8, n_pts, r, method='uniform')
        n_bins = int(lbp_map.max()) + 1
        hist, _ = np.histogram(lbp_map, bins=n_bins, range=(0, n_bins), density=True)
        features.extend(hist)
    return np.array(features, dtype=np.float32)

def extract_glcm_features(image):
    img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        img_uint8, distances=[1], angles=angles,
        levels=256, symmetric=True, normed=True
    )
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        vals = graycoprops(glcm, prop).flatten()
        features.extend(vals)
    return np.array(features, dtype=np.float32)

def extract_fusion_features(image):
    f_gabor = extract_gabor_features(image)
    f_lbp   = extract_lbp_features(image)
    f_glcm  = extract_glcm_features(image)
    return np.concatenate([f_gabor, f_lbp, f_glcm])

def extract_wavelet_features(image, wavelet='haar', level=2):
    img_float64 = image.astype(np.float64) * 255.0 if image.max() <= 1.0 else image.astype(np.float64)
    coeffs = pywt.wavedec2(img_float64, wavelet=wavelet, level=level)
    all_bands = [coeffs[0]] + [c for tup in coeffs[1:] for c in tup]
    features = []
    for band in all_bands:
        flat = band.ravel()
        mean    = np.mean(flat)
        std     = np.std(flat)
        energy  = np.sum(flat ** 2)
        prob    = (np.abs(flat) + 1e-10) / (np.sum(np.abs(flat)) + 1e-10)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        features.extend([mean, std, energy, entropy])
    return np.array(features, dtype=np.float32)

# UI STYLES
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #666; font-size: 1rem; margin-bottom: 2rem; }
    .result-card {
        border-radius: 16px; padding: 1.5rem 2rem;
        margin: 1rem 0; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .severity-bar {
        display: flex; gap: 6px; margin: 1rem 0;
        justify-content: center;
    }
    .sev-item {
        flex: 1; padding: 6px 4px; border-radius: 8px;
        font-size: 0.72rem; font-weight: 600; text-align: center;
        color: white; opacity: 0.3;
    }
    .sev-item.active { opacity: 1; transform: translateY(-3px); }
    .prob-row {
        display: flex; align-items: center; gap: 10px;
        margin: 6px 0;
    }
    .prob-label { width: 160px; font-size: 0.85rem; font-weight: 500; }
    .prob-bar-bg {
        flex: 1; background: #e9ecef; border-radius: 20px; height: 14px;
    }
    .prob-bar-fill {
        height: 14px; border-radius: 20px;
        transition: width 0.6s ease;
    }
    .prob-pct { width: 50px; text-align: right; font-size: 0.85rem; font-weight: 700; }
    .info-box {
        background: #f8f9fa; border-left: 4px solid #667eea;
        border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0;
        font-size: 0.9rem;
    }
    .step-badge {
        display: inline-block; background: #667eea; color: white;
        border-radius: 50%; width: 24px; height: 24px;
        text-align: center; line-height: 24px; font-size: 0.75rem;
        font-weight: 700; margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<p class="main-title">🧠 Alzheimer MRI Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle"> Upload your brain MRI and get a prediction of your Alzheimers severity </p>', unsafe_allow_html=True)

# LOAD MODEL
with st.spinner("Memuat model..."):
    try:
        svm_model, rf_model, scaler = load_models()
        st.success("The model has been successfully loaded")
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()

# SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_choice = st.radio(
        "Select a Prediction Model",
        ["SVM (Support Vector Machine)", "Random Forest", "Both of them (Compare)"],
        index=2
    )
    st.markdown("---")
    st.markdown("### 📋 Prediction Class")
    for cls in SEVERITY_ORDER:
        label, icon, color = CLASS_LABELS[cls]
        st.markdown(f"{icon} **{label}**")
    st.markdown("---")
    st.markdown("### ℹ️ Pipeline info")
    st.markdown("""
    **Preprocessing:**
    - Resize → Grayscale → Gaussian Blur → CLAHE
    - Active Contour Segmentation (Snake)

    **Feature Extraction:**
    - Gabor (84 filters × 4 desc = 336)
    - LBP (5 radii)
    - GLCM (6 properties × 4 angles = 24)

    **Model:** SVM (RBF) & Random Forest
    """)

# MAIN CONTENT
col_upload, col_result = st.columns([1, 1.4], gap="large")

with col_upload:
    st.markdown("### <span class='step-badge'>1</span> Upload MRI Image", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Select an MRI image file (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI image in JPG or PNG format"
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(pil_img)
        st.image(pil_img, caption="Uploaded MRI images", use_container_width=True)

        st.markdown("### <span class='step-badge'>2</span> Preprocessing", unsafe_allow_html=True)
        with st.spinner("Preprocessing in progress... (please wait)"):
            try:
                norm_img, mask, clahe_img = preprocess_image_array(img_array)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image((clahe_img).astype(np.uint8), caption="After CLAHE", use_container_width=True, clamp=True)
                with col_b:
                    st.image((norm_img * 255).astype(np.uint8), caption="After Snake ROI", use_container_width=True, clamp=True)
                preprocess_ok = True
            except Exception as e:
                st.error(f"Failed Preprocessing : {e}")
                preprocess_ok = False

with col_result:
    st.markdown("### <span class='step-badge'>3</span> Prediction Results", unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
            👈 Upload the MRI image on the left to begin the prediction.
        </div>
        """, unsafe_allow_html=True)

    elif preprocess_ok:
        with st.spinner("Feature extraction and prediction..."):
            try:
                features = extract_fusion_features(norm_img)
                feat_scaled = scaler.transform([features])

                def show_prediction(model, model_name, color_accent):
                    pred_class = CLASS_NAMES[model.predict(feat_scaled)[0]] if hasattr(model, 'predict') else None
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(feat_scaled)[0]
                        pred_idx = np.argmax(proba)
                        pred_class = CLASS_NAMES[pred_idx]
                    else:
                        pred_idx = CLASS_NAMES.index(pred_class)
                        proba = np.zeros(len(CLASS_NAMES))
                        proba[pred_idx] = 1.0

                    label, icon, color = CLASS_LABELS[pred_class]

                    st.markdown(f"""
                    <div class="result-card" style="border: 2px solid {color}; background: {color}10;">
                        <div style="font-size: 2.5rem;">{icon}</div>
                        <div style="font-size: 0.8rem; color: #888; margin-top: 4px;">{model_name}</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: {color}; margin: 6px 0;">{label}</div>
                        <div class="severity-bar">
                    """ + "".join([
                        f'<div class="sev-item {"active" if cls == pred_class else ""}" '
                        f'style="background: {CLASS_LABELS[cls][2]};">'
                        f'{CLASS_LABELS[cls][0].replace(" Demented","").replace("Non ","Non")}</div>'
                        for cls in SEVERITY_ORDER
                    ]) + """
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability bars
                    st.markdown("**Per class Probability:**")
                    for i, cls in enumerate(CLASS_NAMES):
                        lbl, _, clr = CLASS_LABELS[cls]
                        pct = proba[i] * 100
                        st.markdown(f"""
                        <div class="prob-row">
                            <div class="prob-label">{lbl}</div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{clr};"></div>
                            </div>
                            <div class="prob-pct">{pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                if "SVM" in model_choice and "Random" not in model_choice:
                    show_prediction(svm_model, "SVM (RBF Kernel)", "#667eea")

                elif "Random Forest" in model_choice and "Keduanya" not in model_choice:
                    show_prediction(rf_model, "Random Forest", "#28a745")

                else:
                    tab1, tab2 = st.tabs(["🔵 SVM", "🟢 Random Forest"])
                    with tab1:
                        show_prediction(svm_model, "SVM (RBF Kernel)", "#667eea")
                    with tab2:
                        show_prediction(rf_model, "Random Forest", "#28a745")

            except Exception as e:
                st.error(f"Failed prediction: {e}")

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#aaa; font-size:0.8rem;">
    🧠 Alzheimer MRI Classifier - Group 3 Digital Image Processing<br>
    <i>Illona Anindya - Rizqi Aqilah - Halilatunnisa</i>
</div>
""", unsafe_allow_html=True)
