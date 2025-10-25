import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import io
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

# Plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ----------------------------------
# Page config (must be first)
# ----------------------------------
st.set_page_config(
    page_title="Model Evaluation (ROC & Calibration)",
    page_icon="❤️",
    layout="wide"
)

# ----------------------------------
# Constants
# ----------------------------------
DEFAULT_THRESHOLD = 0.08
TARGET_COL = "HeartDisease"

# Built‑in logistic regression coefficients from your statsmodels output
BUILT_IN_COEFFICIENTS = {
    "const": -5.9388,
    "BMI": 0.0154,
    "Sex": 0.7322,
    "AgeCategory": 0.2839,
    "Smoking": 0.4337,
    "AlcoholDrinking": -0.2305,
    "DiffWalking": 0.4101,
    "PhysicalActivity": -0.0791,
    "SleepTime": -0.0331,
    "PhysicalHealth": 0.0212,
    "MentalHealth": 0.0104,
    "Asthma": 0.3485,
    "KidneyDisease": 0.7121,
    "SkinCancer": 0.1120,
    "Stroke": 1.1452,
    "Diabetic": 0.6158,
}

FEATURE_ORDER = [
    "BMI", "Sex", "AgeCategory", "Smoking", "AlcoholDrinking", "DiffWalking",
    "PhysicalActivity", "SleepTime", "PhysicalHealth", "MentalHealth", "Asthma",
    "KidneyDisease", "SkinCancer", "Stroke", "Diabetic"
]

# ----------------------------------
# Utility Functions
# ----------------------------------
def sigmoid(z: float) -> float:
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-z))


def load_pickle_file(file_uploader) -> Optional[object]:
    """
    Safely load a pickle file from Streamlit file uploader.
    Tries pickle first, then joblib as fallback.
    """
    if file_uploader is None:
        return None
    
    try:
        # Read bytes from uploader
        file_bytes = file_uploader.getvalue()
        
        # Try pickle first
        try:
            return pickle.load(io.BytesIO(file_bytes))
        except Exception:
            # Fallback to joblib
            return joblib.load(io.BytesIO(file_bytes))
            
    except Exception as e:
        st.error(f"Failed to load {file_uploader.name}: {str(e)}")
        return None


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to numeric codes"""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype("category").cat.codes
    return out


def compute_youden_threshold(y_true, probs):
    """Compute optimal threshold using Youden's J statistic"""
    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    best = np.argmax(j)
    return float(np.clip(thr[best], 1e-3, 0.999))


def predict_single_case(X_input: dict, mode: str, model, encoder) -> Tuple[float, str]:
    """
    Make prediction for a single case using either built-in coefficients or uploaded model
    Returns: (probability, risk_category)
    """
    df_input = pd.DataFrame([X_input], columns=FEATURE_ORDER)
    
    if mode == "Load a trained scikit‑learn model (.pkl)" and model is not None:
        try:
            X_use = df_input.copy()
            if encoder is not None:
                X_use = encoder.transform(X_use)
            else:
                X_use = ensure_numeric(X_use)
            
            proba = model.predict_proba(X_use)[0, 1]
            return proba, categorize_risk(proba)
            
        except Exception as e:
            st.warning(f"Model prediction failed: {e}. Using built-in coefficients.")
    
    # Built-in coefficient path
    z = BUILT_IN_COEFFICIENTS["const"]
    for feat in FEATURE_ORDER:
        z += BUILT_IN_COEFFICIENTS[feat] * X_input[feat]
    
    proba = sigmoid(z)
    return proba, categorize_risk(proba)


def categorize_risk(proba: float) -> str:
    """Categorize risk level based on probability"""
    if proba < 0.05:
        return "Very Low"
    elif proba < 0.15:
        return "Low"
    elif proba < 0.30:
        return "Moderate"
    elif proba < 0.50:
        return "High"
    else:
        return "Very High"


# ----------------------------------
# Initialize Session State
# ----------------------------------
if "threshold" not in st.session_state:
    st.session_state.threshold = DEFAULT_THRESHOLD
if "_suggest_j" not in st.session_state:
    st.session_state._suggest_j = False

# ----------------------------------
# Branded Header
# ----------------------------------
APP_DIR = Path(__file__).resolve().parent
ICON_PATH = APP_DIR.parent / "images" / "heart-rate.png"

try:
    if ICON_PATH.exists():
        icon_img = Image.open(ICON_PATH)
        col1, col2 = st.columns([0.18, 3])
        with col1:
            st.image(icon_img, width=120)
        with col2:
            st.title("Heart Disease Risk Predictor & Model Evaluation")
    else:
        st.title("🫀 Heart Disease Risk Predictor & Model Evaluation")
except Exception:
    st.title("🫀 Heart Disease Risk Predictor & Model Evaluation")

st.caption(
    """Binary classification using variables identified as most predictive in your analysis.
    Upload a trained model or use the built-in logistic regression coefficients."""
)

# ----------------------------------
# Sidebar: Model Selection
# ----------------------------------
st.sidebar.header("⚙️ Model Selection")

mode = st.sidebar.radio(
    "Prediction backend",
    ["Built‑in Logistic Regression (coefficients)", "Load a trained scikit‑learn model (.pkl)"],
    index=0
)

model = None
encoder = None

if mode == "Load a trained scikit‑learn model (.pkl)":
    st.sidebar.subheader("📁 Upload Model Files")
    uploaded_model = st.sidebar.file_uploader(
        "Upload trained model (.pkl)", 
        type=["pkl"],
        key="model_uploader"
    )
    uploaded_encoder = st.sidebar.file_uploader(
        "(Optional) Upload encoder/transformer (.pkl)", 
        type=["pkl"],
        key="encoder_uploader"
    )
    
    # Load the model and encoder
    if uploaded_model:
        model = load_pickle_file(uploaded_model)
        if model is not None:
            st.sidebar.success(f"✅ Model loaded: {uploaded_model.name}")
    
    if uploaded_encoder:
        encoder = load_pickle_file(uploaded_encoder)
        if encoder is not None:
            st.sidebar.success(f"✅ Encoder loaded: {uploaded_encoder.name}")

# ----------------------------------
# Sidebar: Threshold Control
# ----------------------------------
st.sidebar.header("🔧 Classification Threshold")
threshold = st.sidebar.slider(
    "Threshold (for classifying Positive)",
    min_value=0.05,
    max_value=0.95,
    value=st.session_state.threshold,
    step=0.01,
    key="threshold_slider"
)
st.session_state.threshold = threshold
st.sidebar.caption("Used for classification, sensitivity/specificity, and confusion matrix.")

# ----------------------------------
# Main Content: Feature Inputs
# ----------------------------------
st.markdown("---")
st.header("📝 Single Case Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    sex = st.selectbox("Sex", ["Female", "Male"])
    age_label_to_code = {
        "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4, "40–44": 5, "45–49": 6,
        "50–54": 7, "55–59": 8, "60–64": 9, "65–69": 10, "70–74": 11, "75–79": 12, "80+": 13
    }
    age_label = st.selectbox("Age Category", list(age_label_to_code.keys()), index=8)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=28.0, step=0.1)
    sleep_time = st.number_input("Average Sleep Time (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

with col2:
    smoking = st.selectbox("Currently Smokes?", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Drinking?", ["No", "Yes"])
    physical_activity = st.selectbox("Physical Activity?", ["No", "Yes"])
    diff_walking = st.selectbox("Difficulty Walking?", ["No", "Yes"])

with col3:
    physical_health = st.number_input("Physical Health (days not good / 30)", min_value=0, max_value=30, value=3)
    mental_health = st.number_input("Mental Health (days not good / 30)", min_value=0, max_value=30, value=3)
    asthma = st.selectbox("Asthma?", ["No", "Yes"])
    kidney = st.selectbox("Kidney Disease?", ["No", "Yes"])
    skin_cancer = st.selectbox("Skin Cancer?", ["No", "Yes"])
    stroke = st.selectbox("History of Stroke?", ["No", "Yes"])
    diabetic = st.selectbox("Diabetic?", ["No", "Yes"])

# Build input dictionary
X_input = {
    "BMI": float(bmi),
    "Sex": 1 if sex == "Male" else 0,
    "AgeCategory": age_label_to_code[age_label],
    "Smoking": 1 if smoking == "Yes" else 0,
    "AlcoholDrinking": 1 if alcohol == "Yes" else 0,
    "DiffWalking": 1 if diff_walking == "Yes" else 0,
    "PhysicalActivity": 1 if physical_activity == "Yes" else 0,
    "SleepTime": float(sleep_time),
    "PhysicalHealth": int(physical_health),
    "MentalHealth": int(mental_health),
    "Asthma": 1 if asthma == "Yes" else 0,
    "KidneyDisease": 1 if kidney == "Yes" else 0,
    "SkinCancer": 1 if skin_cancer == "Yes" else 0,
    "Stroke": 1 if stroke == "Yes" else 0,
    "Diabetic": 1 if diabetic == "Yes" else 0,
}

# Prediction button
if st.button("🔮 Predict Heart Disease Risk", type="primary"):
    proba, risk_category = predict_single_case(X_input, mode, model, encoder)
    
    # Display results
    st.markdown("### 📊 Prediction Results")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Risk Probability", f"{proba:.1%}")
    with res_col2:
        st.metric("Risk Category", risk_category)
    
    # Risk interpretation
    if proba < threshold:
        st.success(f"✅ **Low Risk**: Below classification threshold ({threshold:.2f})")
    else:
        st.error(f"⚠️ **High Risk**: Above classification threshold ({threshold:.2f})")
    
    # Visual probability bar
    fig_bar, ax_bar = plt.subplots(figsize=(8, 1.5), dpi=100)
    ax_bar.barh([0], [proba], color='crimson', alpha=0.7, height=0.5)
    ax_bar.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Probability of Heart Disease")
    ax_bar.set_yticks([])
    ax_bar.legend()
    ax_bar.grid(axis='x', alpha=0.3)
    st.pyplot(fig_bar)

# ----------------------------------
# Model Evaluation Section
# ----------------------------------
st.markdown("---")
st.header("📈 Model Evaluation (ROC & Calibration)")
st.caption(
    "Upload a CSV with feature columns and a binary 'HeartDisease' column "
    "to compute ROC, AUC, calibration, and a confusion matrix at the selected threshold."
)

# Display current model
st.markdown(
    f"""<div style="padding:.5rem 1rem;border:1px solid #444;border-radius:8px;
    display:inline-block;margin:.5rem 0;background-color:#1e1e1e;">
    <b>Model in use:</b> {mode}
    </div>""",
    unsafe_allow_html=True
)

# Threshold controls
c1, c2 = st.columns([1, 1])
with c1:
    if st.button("✨ Suggest Best Threshold (Youden's J)"):
        st.session_state._suggest_j = True
with c2:
    if st.button("🔄 Reset to Default (0.08)"):
        st.session_state.threshold = DEFAULT_THRESHOLD
        st.rerun()

# File upload for evaluation
eval_file = st.file_uploader(
    "📄 Upload evaluation CSV",
    type=["csv"],
    help="CSV must include feature columns and a 'HeartDisease' target column"
)

# Main evaluation logic
if eval_file is not None:
    try:
        df = pd.read_csv(eval_file)
        
        if TARGET_COL not in df.columns:
            st.error(f"❌ CSV must include the binary target column '{TARGET_COL}'.")
        else:
            y_true = df[TARGET_COL].astype(int).values
            X = df.drop(columns=[TARGET_COL])
            
            backend_used = mode
            proba = None
            
            # Try uploaded model first if selected
            if mode.startswith("Load") and model is not None:
                try:
                    X_use = X.copy()
                    if encoder is not None:
                        X_use = encoder.transform(X_use)
                    else:
                        X_use = ensure_numeric(X_use)
                    proba = model.predict_proba(X_use)[:, 1]
                    st.info(f"✅ Using uploaded model: {uploaded_model.name}")
                except Exception as e:
                    st.warning(f"⚠️ Uploaded model failed to predict ({e}). Using built-in Logistic Regression instead.")
                    backend_used = "Built-in Logistic Regression (coefficients)"
                    proba = None
            
            # Built-in LR path (with cross-validation to reduce bias)
            if proba is None:
                X_enc = ensure_numeric(X)
                try:
                    with st.spinner("Training built-in Logistic Regression with cross-validation..."):
                        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        blr = LogisticRegression(max_iter=200, random_state=42)
                        proba = cross_val_predict(blr, X_enc, y_true, cv=skf, method="predict_proba")[:, 1]
                    st.info("✅ Using built-in Logistic Regression (5-fold CV)")
                except Exception as e:
                    st.error(f"❌ Built-in Logistic Regression failed: {e}")
                    proba = np.zeros_like(y_true, dtype=float)
            
            # Handle Youden's J suggestion
            if st.session_state._suggest_j:
                try:
                    suggested_thr = round(compute_youden_threshold(y_true, proba), 3)
                    st.session_state.threshold = suggested_thr
                    st.success(f"✅ Optimal threshold set to: {suggested_thr:.3f} (Youden's J)")
                    st.session_state._suggest_j = False
                    st.rerun()
                except Exception as e:
                    st.warning(f"⚠️ Could not compute Youden's J: {e}")
                    st.session_state._suggest_j = False
            
            thr = float(st.session_state.threshold)
            y_pred = (proba >= thr).astype(int)
            
            # --- Compute Metrics ---
            auc = roc_auc_score(y_true, proba)
            acc = accuracy_score(y_true, y_pred)
            sens = recall_score(y_true, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Display metrics
            st.markdown("### 📊 Performance Metrics")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("AUC", f"{auc:.3f}")
            k2.metric("Sensitivity (TPR)", f"{sens:.3f}")
            k3.metric("Specificity (TNR)", f"{spec:.3f}")
            k4.metric("Accuracy", f"{acc:.3f}")
            
            # --- ROC Curve ---
            st.markdown("### 📈 ROC Curve")
            fpr, tpr, _ = roc_curve(y_true, proba)
            fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=120)
            ax1.plot(fpr, tpr, lw=2.5, label=f'ROC (AUC = {auc:.3f})', color='#2E86AB')
            ax1.plot([0, 1], [0, 1], '--', lw=1.5, color='gray', label='Random Classifier')
            ax1.set_xlabel("False Positive Rate", fontsize=11)
            ax1.set_ylabel("True Positive Rate", fontsize=11)
            ax1.set_title(f"ROC Curve - {backend_used}", fontsize=12, fontweight='bold')
            ax1.legend(loc='lower right')
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)
            
            # --- Calibration Curve ---
            st.markdown("### 📉 Calibration Curve")
            prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10, strategy='uniform')
            fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=120)
            ax2.plot(prob_pred, prob_true, marker='o', lw=2, markersize=8, color='#A23B72', label='Model Calibration')
            ax2.plot([0, 1], [0, 1], '--', lw=1.5, color='gray', label='Perfect Calibration')
            ax2.set_xlabel("Predicted Probability", fontsize=11)
            ax2.set_ylabel("Observed Frequency", fontsize=11)
            ax2.set_title("Calibration Plot", fontsize=12, fontweight='bold')
            ax2.legend(loc='upper left')
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)
            
            # --- Confusion Matrix ---
            st.markdown("### 🔢 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative (0)", "Actual Positive (1)"],
                columns=["Predicted Negative (0)", "Predicted Positive (1)"]
            )
            st.dataframe(cm_df, use_container_width=True)
            
            # Detailed breakdown
            with st.expander("📋 See Detailed Breakdown"):
                st.write(f"**True Negatives (TN):** {tn}")
                st.write(f"**False Positives (FP):** {fp}")
                st.write(f"**False Negatives (FN):** {fn}")
                st.write(f"**True Positives (TP):** {tp}")
                
    except Exception as e:
        st.error(f"❌ Error processing evaluation file: {str(e)}")
else:
    st.info("📁 Upload a CSV file to begin model evaluation.")

# ----------------------------------
# Footer & Documentation
# ----------------------------------
st.markdown("---")
st.markdown("### 📚 Notes")
st.markdown(
    """
    - **Built-in Logistic Regression**: Uses coefficients from your statsmodels output (post-VIF fix) for single predictions. 
      For evaluation, it trains a scikit-learn LogisticRegression with 5-fold cross-validation to reduce bias.
    - **Uploaded Model**: Use the sidebar to upload a trained scikit-learn model (.pkl file). Optionally upload an encoder/transformer.
    - **Threshold Optimization**: Click "Suggest Best Threshold" to compute the optimal threshold using Youden's J statistic 
      (maximizes sensitivity + specificity - 1).
    - **Model Switching**: You can freely switch between built-in and uploaded models. The app will reload models as needed.
    """
)

st.markdown("---")
st.markdown("#### ⚠️ Disclaimer")
st.markdown("*This application is intended for educational purposes only. It does not offer medical advice, diagnosis, or treatment.*")

# ----------------------------------
# Contact Section
# ----------------------------------
st.markdown("---")
st.markdown("### 📬 Contact Us")
with st.form("contact_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Your Message")
    contact_submit = st.form_submit_button("Send Message")
    if contact_submit:
        if name and email and message:
            st.success("✅ Thank you! Your message has been received.")
        else:
            st.warning("⚠️ Please fill in all fields.")

st.markdown("---")
st.markdown("© 2025 William C. Phiri – Powered by BEDA | Email: wphiri@beda.ie")