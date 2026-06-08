import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# PAGE CONFIG
st.set_page_config(
    page_title="Model Visualization",
    page_icon="📊",
    layout="wide",
)

# CONSTANTS
CLASS_NAMES   = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
CLASS_LABELS  = ["Non\nDemented", "Very Mild\nDemented", "Mild\nDemented", "Moderate\nDemented"]
CLASS_COLORS  = ["#28a745", "#ffc107", "#fd7e14", "#dc3545"]

# STYLES
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #11998e, #38ef7d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #888; font-size: 1rem; margin-bottom: 2rem; }
    .metric-card {
        background: #1e1e2e; border-radius: 14px;
        padding: 1.2rem 1.5rem; text-align: center;
        border: 1px solid #333;
    }
    .metric-val { font-size: 2rem; font-weight: 800; }
    .metric-lbl { font-size: 0.8rem; color: #888; margin-top: 2px; }
    .section-title {
        font-size: 1.2rem; font-weight: 700;
        border-left: 4px solid #38ef7d;
        padding-left: 10px; margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<p class="main-title">📊 Model Performance Visualization</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A comprehensive analysis of the evaluation results for the SVM and Random Forest models on the Alzheimers MRI dataset.</p>', unsafe_allow_html=True)

# LOAD DATA
@st.cache_resource
def load_eval_data():
    return joblib.load("results/eval_data.pkl")

try:
    data = load_eval_data()
except FileNotFoundError:
    st.error("File `results/eval_data.pkl` not found!")
    st.stop()

y_test      = np.array(data['y_test'])
y_pred_svm  = np.array(data['y_pred_svm'])
y_pred_rf   = np.array(data['y_pred_rf'])
acc_svm     = data['acc_svm']
acc_rf      = data['acc_rf']
f1_svm      = data['f1_svm']
f1_rf       = data['f1_rf']
X_test_pca  = np.array(data['X_test_pca'])
X_tsne      = np.array(data['X_tsne'])
shap_values = data['shap_values']
feat_names  = data['feature_names']

# SECTION 1: METRICS OVERVIEW
st.markdown('<div class="section-title">🏆 Model Performance Comparison</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="color:#667eea;">{acc_svm*100:.1f}%</div>
        <div class="metric-lbl">SVM Accuracy</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="color:#667eea;">{f1_svm*100:.1f}%</div>
        <div class="metric-lbl">SVM F1-Score</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="color:#28a745;">{acc_rf*100:.1f}%</div>
        <div class="metric-lbl">RF Accuracy</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="color:#28a745;">{f1_rf*100:.1f}%</div>
        <div class="metric-lbl">RF F1-Score</div>
    </div>""", unsafe_allow_html=True)

# SECTION 2: CONFUSION MATRIX
st.markdown('<div class="section-title">🔲 Confusion Matrix</div>', unsafe_allow_html=True)

model_choice = st.radio("Select a Model:", ["SVM", "Random Forest", "Both"], horizontal=True)

def plot_confusion_matrix(y_true, y_pred, title, color):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.heatmap(
        cm_norm, annot=cm, fmt='d',
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        cmap=sns.light_palette(color, as_cmap=True),
        linewidths=0.5, linecolor='#333',
        ax=ax, annot_kws={"size": 13, "weight": "bold", "color": "white"}
    )
    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted', color='white', fontsize=10)
    ax.set_ylabel('Actual', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    plt.tight_layout()
    return fig

if model_choice == "SVM":
    fig = plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix - SVM", "#667eea")
    st.pyplot(fig)
elif model_choice == "Random Forest":
    fig = plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix - Random Forest", "#28a745")
    st.pyplot(fig)
else:
    c1, c2 = st.columns(2)
    with c1:
        fig = plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix - SVM", "#667eea")
        st.pyplot(fig)
    with c2:
        fig = plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix - Random Forest", "#28a745")
        st.pyplot(fig)

# SECTION 3: CLASSIFICATION REPORT
st.markdown('<div class="section-title">📋 Classification Report</div>', unsafe_allow_html=True)

tab_svm, tab_rf = st.tabs(["🔵 SVM", "🟢 Random Forest"])

def report_to_df(y_true, y_pred):
    import pandas as pd
    report = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True)
    df = pd.DataFrame(report).T
    df = df.drop(['macro avg', 'weighted avg'], errors='ignore')
    df = df[['precision', 'recall', 'f1-score', 'support']]
    df['support'] = df['support'].astype(int)
    df = df.round(3)
    return df

with tab_svm:
    df_svm = report_to_df(y_test, y_pred_svm)
    st.dataframe(df_svm.style.background_gradient(cmap='Blues', subset=['precision','recall','f1-score']), use_container_width=True)

with tab_rf:
    df_rf = report_to_df(y_test, y_pred_rf)
    st.dataframe(df_rf.style.background_gradient(cmap='Greens', subset=['precision','recall','f1-score']), use_container_width=True)

# SECTION 4: PCA & t-SNE
st.markdown('<div class="section-title">🔵 PCA & t-SNE - Feature Space Distribution</div>', unsafe_allow_html=True)

col_pca, col_tsne = st.columns(2)

def scatter_plot(X, y, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    for i, (cls, color) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=color, label=cls.replace('\n', ' '),
                   alpha=0.6, s=15, edgecolors='none')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')
    plt.tight_layout()
    return fig

with col_pca:
    if X_test_pca.shape[1] >= 2:
        fig = scatter_plot(X_test_pca, y_test, "PCA - 2 Main Components")
        st.pyplot(fig)
    else:
        st.warning("PCA data is not available")

with col_tsne:
    if X_tsne.shape[1] >= 2:
        y_tsne = np.array(y_test[:len(X_tsne)])
        fig = scatter_plot(X_tsne, y_tsne, "t-SNE - 2D Embedding")
    else:
        st.warning("t-SNE data is not available")

# SECTION 5: SHAP FEATURE IMPORTANCE
st.markdown('<div class="section-title">🔍 SHAP - Feature Importance</div>', unsafe_allow_html=True)

try:
    shap_arr = np.array(shap_values)

    if shap_arr.ndim == 3:
        mean_shap = np.abs(shap_arr).mean(axis=(0, 1))
    elif shap_arr.ndim == 2:
        mean_shap = np.abs(shap_arr).mean(axis=0)
    else:
        mean_shap = np.abs(shap_arr)

    feat_names_arr = np.array(feat_names) if feat_names is not None else np.arange(len(mean_shap))

    top_n = st.slider("Tampilkan Top N Fitur:", min_value=10, max_value=min(50, len(mean_shap)), value=20)

    top_idx   = np.argsort(mean_shap)[-top_n:][::-1]
    top_vals  = mean_shap[top_idx]
    top_names = feat_names_arr[top_idx]

    fig, ax = plt.subplots(figsize=(10, top_n * 0.35 + 1))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_n))
    bars = ax.barh(range(top_n), top_vals[::-1], color=colors[::-1], edgecolor='none', height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=8, color='white')
    ax.set_xlabel('Mean |SHAP Value|', color='white', fontsize=10)
    ax.set_title(f'Top {top_n} Feature Importance (SHAP)', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.warning(f"SHAP cannot be displayed: {e}")

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#aaa; font-size:0.8rem;">
    🧠 Alzheimer MRI Classifier - Group 3 Digital Image Processing<br>
    <i>Illona Anindya - Rizqi Aqilah - Halilatunnisa</i>
</div>
""", unsafe_allow_html=True)
