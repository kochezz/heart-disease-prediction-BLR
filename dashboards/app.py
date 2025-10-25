
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

st.set_page_config(page_title="Model Evaluation (ROC & Calibration)", page_icon="❤️", layout="wide")

# ------------------- helpers -------------------
DEFAULT_THRESHOLD = 0.08  # matches your original default
TARGET_COL = "HeartDisease"
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Tuple
import io
from PIL import Image
import smtplib, ssl
from email.message import EmailMessage

# Plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, accuracy_score,
    precision_score, recall_score
)
from sklearn.calibration import calibration_curve

# ----------------------------------
# Page config + icon path (robust)
# ----------------------------------
APP_DIR = Path(__file__).resolve().parent
ICON_PATH = APP_DIR.parent / "images" / "heart-rate.png"  
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon=str(ICON_PATH),  # favicon
    layout="wide"
)

# ----------------------------------
# Branded header (image + title)
# ----------------------------------
try:
    icon_img = Image.open(ICON_PATH)
    col1, col2 = st.columns([0.18, 3])
    with col1:
        st.image(icon_img, width=120) 
    with col2:
        st.title("Heart Disease Risk Predictor")
except Exception as e:
    # Graceful fallback if icon can't be loaded
    st.title("🫀 Heart Disease Risk Predictor")
    st.caption("Note: custom icon could not be loaded. Error: {}".format(e))

st.caption(
    """Binary classification using variables identified as most predictive in your analysis.
If a trained model is not provided, the app uses built‑in logistic regression coefficients from your statsmodels output (post‑VIF fix)."""
)

# ------------------------------
# Utility
# ------------------------------
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def load_model_artifacts(model_bytes: Optional[bytes], encoder_bytes: Optional[bytes]) -> Tuple[Optional[object], Optional[object]]:
    model = None
    encoder = None
    try:
        if model_bytes is not None:
            model = pickle.load(io.BytesIO(model_bytes))
    except Exception as e:
        st.warning(f"Could not load model: {e}")
    try:
        if encoder_bytes is not None:
            encoder = pickle.load(io.BytesIO(encoder_bytes))
    except Exception as e:
        st.warning(f"Could not load encoder: {e}")
    return model, encoder


# ------------------------------
# Sidebar Controls
# ------------------------------

st.sidebar.header("⚙️ App Settings")
mode = st.sidebar.radio(
    "Prediction backend",
    ["Built‑in Logistic Regression (coefficients)", "Load a trained scikit‑learn model (.pkl)"]
)

uploaded_model = None
uploaded_encoder = None
if mode == "Load a trained scikit‑learn model (.pkl)":
    uploaded_model = st.sidebar.file_uploader("Upload trained model (.pkl)", type=["pkl"])
    uploaded_encoder = st.sidebar.file_uploader("(Optional) Upload encoder/transformer (.pkl)", type=["pkl"])

# Threshold slider
st.sidebar.subheader("🔧 Classification Threshold")
threshold = st.sidebar.slider("Threshold (for classifying Positive)", min_value=0.05, max_value=0.95, value=0.08, step=0.01)
st.sidebar.caption("Used for classification, sensitivity/specificity, and confusion matrix.")

# Evaluation data uploader
st.sidebar.subheader("📄 Evaluation Dataset (optional)")
eval_csv = st.sidebar.file_uploader("Upload CSV with features + 'HeartDisease' column for ROC/Calibration", type=["csv"])

# ------------------------------
# Feature Inputs
# ------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    sex = st.selectbox("Sex", ["Female", "Male"])
    age_label_to_code = {
        "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4, "40–44": 5, "45–49": 6,
        "50–54": 7, "55–59": 8, "60–64": 9, "65–69": 10, "70–74": 11, "75–79": 12, "80+": 13
    }
    age_label = st.selectbox("Age Category", list(age_label_to_code.keys()), index=8)  # default ~60–64
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

# Numeric encodings consistent with your modeling
X_input = {
    "BMI": float(bmi),
    "Sex": 1 if sex == "Male" else 0,
    "AgeCategory": age_label_to_code[age_label],  # ordinal encoding used during modeling
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

feature_order = [
    "BMI","Sex","AgeCategory","Smoking","AlcoholDrinking","DiffWalking",
    "PhysicalActivity","SleepTime","PhysicalHealth","MentalHealth","Asthma",
    "KidneyDisease","SkinCancer","Stroke","Diabetic"
]

# Built‑in logistic regression coefficients from your statsmodels output (03_modeling, post‑VIF fix)
coeffs = {
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

# ------------------------------
# Single Prediction
# ------------------------------

st.markdown("---")
predict_btn = st.button("🔮 Predict Heart Disease Risk (Single Case)")

if predict_btn:
    df_input = pd.DataFrame([X_input], columns=feature_order)

    if mode == "Load a trained scikit‑learn model (.pkl)" and uploaded_model is not None:
        model, encoder = load_model_artifacts(uploaded_model.read(), uploaded_encoder.read() if uploaded_encoder else None)

        if model is None:
            st.error("Model could not be loaded. Falling back to built‑in BLR.")
        else:
            if encoder is not None:
                try:
                    X_transformed = encoder.transform(df_input)
                except Exception:
                    X_transformed = encoder.transform(df_input.values)
            else:
                X_transformed = df_input.values

            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X_transformed)[:, 1][0])
            elif hasattr(model, "decision_function"):
                z = float(model.decision_function(X_transformed)[0])
                proba = sigmoid(z)
            else:
                pred = int(model.predict(X_transformed)[0])
                proba = float(pred)

            backend = "Uploaded scikit‑learn model"
            contributions = None
    
    if mode == "Built‑in Logistic Regression (coefficients)" or (predict_btn and ('backend' not in locals())):
        # Built‑in BLR
        z = coeffs["const"]
        contributions = {}
        for k in feature_order:
            c = coeffs[k] * float(df_input.iloc[0][k])
            contributions[k] = c
            z += c
        proba = float(sigmoid(z))
        backend = "Built‑in Logistic Regression (coefficients)"

    # Display
    st.subheader("Prediction Result")
    st.write(f"**Backend:** {backend}")

    risk_pct = 100.0 * proba
    pred_label = 1 if proba >= threshold else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Probability", f"{risk_pct:.1f}%")
    c2.metric("Risk Band", "Low" if risk_pct < 7 else ("Moderate" if risk_pct < 15 else "High"))
    c3.metric("Applied Threshold", f"{threshold:.2f}")

    st.progress(min(1.0, proba))

    if contributions is not None:
        st.markdown("### Feature Contributions (log‑odds)")
        contrib_df = pd.DataFrame({
            "Feature": list(contributions.keys()),
            "Contribution": list(contributions.values())
        }).sort_values("Contribution", ascending=False)
        st.dataframe(contrib_df, use_container_width=True)
        st.caption("Positive values increase risk (log‑odds), negative values decrease risk. Based on BLR coefficients.")

    with st.expander("Show input vector"):
        st.json(X_input)

# ------------------------------
# Evaluation: ROC, Calibration, Confusion Matrix (Requires Dataset)
# ------------------------------

st.markdown("---")
st.header("Model Evaluation (ROC & Calibration)")
st.caption("Upload a CSV with feature columns and a binary 'HeartDisease' column to compute ROC, AUC, calibration, and a confusion matrix at the selected threshold.")

# Quick helper to compute Youden's J on the uploaded CSV and move the slider suggestion
col_tools = st.columns([1,1,2])
with col_tools[0]:
    compute_best = st.button("📐 Suggest Best Threshold (Youden's J)")
with col_tools[1]:
    reset_default = st.button("↩️ Reset to Default (0.08)")

if eval_csv is not None:
    import io
    eval_df = pd.read_csv(eval_csv)

    missing = [c for c in feature_order + ["HeartDisease"] if c not in eval_df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
    else:
        X_eval = eval_df[feature_order].copy()
        y_true = eval_df["HeartDisease"].astype(int).values

        # Get probabilities from chosen backend
        proba_eval = None
        backend_used = None

        if mode == "Load a trained scikit‑learn model (.pkl)" and uploaded_model is not None:
            model, encoder = load_model_artifacts(uploaded_model.read(), uploaded_encoder.read() if uploaded_encoder else None)
            if model is not None:
                if encoder is not None:
                    try:
                        X_transformed = encoder.transform(X_eval)
                    except Exception:
                        X_transformed = encoder.transform(X_eval.values)
                else:
                    X_transformed = X_eval.values

                if hasattr(model, "predict_proba"):
                    proba_eval = model.predict_proba(X_transformed)[:, 1]
                elif hasattr(model, "decision_function"):
                    z = model.decision_function(X_transformed)
                    proba_eval = 1.0 / (1.0 + np.exp(-z))
                else:
                    proba_eval = model.predict(X_transformed).astype(float)
                backend_used = "Uploaded scikit‑learn model"

        if proba_eval is None:
            # Fallback to built-in BLR
            z = np.full(len(X_eval), coeffs["const"], dtype=float)
            for k in feature_order:
                z += coeffs[k] * X_eval[k].astype(float).values
            proba_eval = 1.0 / (1.0 + np.exp(-z))
            backend_used = "Built‑in Logistic Regression (coefficients)"

        # Metrics
        from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score
        from sklearn.calibration import calibration_curve
        fpr, tpr, thr = roc_curve(y_true, proba_eval)
        auc_val = roc_auc_score(y_true, proba_eval)

        # Optional: suggest best threshold
        if compute_best:
            youden_j = tpr - fpr
            best_idx = int(np.argmax(youden_j))
            best_thr = float(thr[best_idx])
            st.success(f"Suggested threshold (Youden's J): {best_thr:.3f}")
        if reset_default:
            st.info("Default threshold is 0.08 (from master dataset).")

        # Threshold-based metrics
        y_pred = (proba_eval >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        st.subheader("Evaluation Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", f"{auc_val:.3f}")
        c2.metric("Sensitivity (TPR)", f"{sensitivity:.3f}")
        c3.metric("Specificity (TNR)", f"{specificity:.3f}")
        c4.metric("Accuracy", f"{accuracy:.3f}")

        # ROC Plot
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.plot(fpr, tpr, lw=2)
        ax1.plot([0, 1], [0, 1], linestyle='--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve ({backend_used})\\nAUC = {auc_val:.3f}')
        st.pyplot(fig1)

        # Calibration Plot
        prob_true, prob_pred = calibration_curve(y_true, proba_eval, n_bins=10, strategy='quantile')
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(prob_pred, prob_true, marker='o', lw=1)
        ax2.plot([0, 1], [0, 1], linestyle='--')
        ax2.set_xlabel('Mean Predicted Probability')
        ax2.set_ylabel('Fraction of Positives')
        ax2.set_title('Calibration Curve')
        st.pyplot(fig2)

        # Confusion Matrix Table
        st.markdown("### Confusion Matrix (at selected threshold)")
        cm_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=pd.Index(["Actual 0", "Actual 1"], name=""),
            columns=["Pred 0", "Pred 1"]
        )
        st.dataframe(cm_df)

else:
    st.info("To generate ROC & Calibration plots, upload an evaluation CSV in the sidebar.")

# ------------------------------
# Model Notes
# ------------------------------

st.markdown("---")
st.markdown(
    """
    **Notes**  
    • Encodings: Binary variables use 0 = No, 1 = Yes. Sex: Male = 1.  
    • AgeCategory uses an ordinal scale (1–13) consistent with BRFSS groups. Adjust the mapping if your trained model differs.  
    • For uploaded models, include any preprocessing pipeline (e.g., StandardScaler, OneHotEncoder) as a single `Pipeline` object, or upload the transformer separately.  
    • The built‑in logistic regression path uses coefficients extracted from your `statsmodels` output **after the multicollinearity (VIF) fix** (offending variable removed).  
    • Use the threshold slider (left sidebar) to tune sensitivity vs. specificity. Consider optimizing via Youden's J on your validation data.
    """
)

# Disclaimer section
st.markdown("#### ⚠️ Disclaimer")
st.markdown("*This application is intended for educational purposes only. It does not offer medical advice, diagnosis or treatment.*")

# ------------------------------
# Contact section (Gmail SMTP)
# ------------------------------
# Contact section
st.markdown("---")
st.markdown("### 📬 Contact Us")
with st.form("contact_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Your Message")
    contact_submit = st.form_submit_button("Send Message")
    if contact_submit:
        st.success("Thank you! Your message has been received.")

st.markdown("---")
st.markdown("© 2025 William C. Phiri – Powered by BEDA | Email: wphiri@beda.ie")

def _load_pickle(file):
    try:
        return pickle.load(file)
    except Exception:
        file.seek(0)
        import joblib
        return joblib.load(file)

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype("category").cat.codes
    return out

def compute_youden_threshold(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    best = np.argmax(j)
    return float(np.clip(thr[best], 1e-3, 0.999))

# ------------------ session defaults ------------------
if "backend_label" not in st.session_state:
    st.session_state.backend_label = "Built-in Logistic Regression (coefficients)"
if "threshold" not in st.session_state:
    st.session_state.threshold = DEFAULT_THRESHOLD

# ------------------ sidebar (model selection) ------------------
st.sidebar.header("Model selection")
mode = st.sidebar.radio(
    "Prediction backend",
    ["Built-in Logistic Regression (coefficients)", "Load a trained model (.pkl)"],
    index=0
)
st.session_state.backend_label = mode

uploaded_model = uploaded_encoder = None
model = encoder = None
if mode.startswith("Load"):
    uploaded_model = st.sidebar.file_uploader("Upload model (.pkl)", type=["pkl"])
    uploaded_encoder = st.sidebar.file_uploader("(Optional) Upload encoder/transformer (.pkl)", type=["pkl"])
    if uploaded_model:
        try:
            model = _load_pickle(uploaded_model)
            st.sidebar.success(f"Model loaded: {uploaded_model.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
    if uploaded_encoder:
        try:
            encoder = _load_pickle(uploaded_encoder)
            st.sidebar.info(f"Transformer loaded: {uploaded_encoder.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load transformer: {e}")

st.sidebar.header("Decision threshold")
st.sidebar.slider("Threshold", 0.0, 1.0, key="threshold", value=st.session_state.threshold, step=0.01)

# ------------------ header ------------------
st.title("Model Evaluation (ROC & Calibration)")
st.caption("Upload a CSV with feature columns and a binary 'HeartDisease' column to compute ROC, AUC, calibration, and a confusion matrix at the selected threshold.")

st.markdown(
    f"""<div style="padding:.5rem 1rem;border:1px solid #444;border-radius:8px;display:inline-block;margin:.2rem 0;">
    <b>Model in use:</b> {st.session_state.backend_label}
    </div>""",
    unsafe_allow_html=True
)

# ------------------ controls ------------------
c1, c2 = st.columns([1,1])
with c1:
    if st.button("Suggest Best Threshold (Youden's J)"):
        st.session_state._suggest_j = True
with c2:
    if st.button("Reset to Default (0.08)"):
        st.session_state.threshold = DEFAULT_THRESHOLD

st.markdown("### Evaluation Summary")
eval_file = st.file_uploader("Upload evaluation CSV", type=["csv"])

# ------------------ main evaluation ------------------
if eval_file is not None:
    df = pd.read_csv(eval_file)
    if TARGET_COL not in df.columns:
        st.error(f"CSV must include the binary target column '{TARGET_COL}'.")
    else:
        y_true = df[TARGET_COL].astype(int).values
        X = df.drop(columns=[TARGET_COL])

        backend_used = st.session_state.backend_label
        proba = None

        # Try uploaded model first if selected
        if backend_used.startswith("Load"):
            if model is None:
                st.error("Uploaded model unavailable; using built-in Logistic Regression instead.")
                backend_used = "Built-in Logistic Regression (coefficients)"
            else:
                try:
                    X_use = X.copy()
                    if encoder is not None:
                        X_use = encoder.transform(X_use)
                    else:
                        X_use = ensure_numeric(X_use)
                    proba = model.predict_proba(X_use)[:,1]
                except Exception as e:
                    st.error(f"Uploaded model failed to predict ({e}); using built-in Logistic Regression instead.")
                    backend_used = "Built-in Logistic Regression (coefficients)"
                    proba = None

        # Built-in LR path
        if backend_used.startswith("Built-in") or proba is None:
            X_enc = ensure_numeric(X)
            try:
                # out-of-fold probabilities to reduce bias
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                blr = LogisticRegression(max_iter=200)
                proba = cross_val_predict(blr, X_enc, y_true, cv=skf, method="predict_proba")[:,1]
            except Exception as e:
                st.error(f"Built-in Logistic Regression failed: {e}")
                proba = np.zeros_like(y_true, dtype=float)

        st.caption(f"Evaluation computed with: **{backend_used}**")

        # Threshold helpers that depend on proba
        if "_suggest_j" in st.session_state and st.session_state._suggest_j:
            try:
                st.session_state.threshold = round(compute_youden_threshold(y_true, proba), 3)
                st.success(f"Set threshold to Youden’s J: {st.session_state.threshold:.3f}")
            except Exception as e:
                st.warning(f"Could not compute Youden’s J: {e}")
            st.session_state._suggest_j = False

        thr = float(st.session_state.threshold)
        y_pred = (proba >= thr).astype(int)

        # --- metrics
        auc = roc_auc_score(y_true, proba)
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("AUC", f"{auc:.3f}")
        k2.metric("Sensitivity (TPR)", f"{sens:.3f}")
        k3.metric("Specificity (TNR)", f"{spec:.3f}")
        k4.metric("Accuracy", f"{acc:.3f}")

        # --- ROC
        st.markdown(f"### ROC Curve ({backend_used})")
        fpr, tpr, _ = roc_curve(y_true, proba)
        fig1, ax1 = plt.subplots(figsize=(6,5), dpi=150)
        ax1.plot(fpr, tpr, lw=2)
        ax1.plot([0,1],[0,1],'--', lw=1)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title(f"AUC = {auc:.3f}")
        st.pyplot(fig1)

        # --- Calibration
        st.markdown("### Calibration Curve")
        prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10, strategy='uniform')
        fig2, ax2 = plt.subplots(figsize=(6,5), dpi=150)
        ax2.plot(prob_pred, prob_true, marker='o', lw=1.5)
        ax2.plot([0,1],[0,1],'--', lw=1)
        ax2.set_xlabel("Predicted probability")
        ax2.set_ylabel("Observed frequency")
        st.pyplot(fig2)

        # --- Confusion matrix (table)
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"])
        st.dataframe(cm_df, use_container_width=True)
else:
    st.info("Upload a CSV to evaluate.")

st.divider()
st.caption("This tool is for educational purposes only and not a substitute for professional medical advice.")
