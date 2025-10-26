import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import io
from pathlib import Path
from typing import Optional, Tuple, Dict
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
APP_DIR = Path(__file__).resolve().parent
ICON_PATH = APP_DIR.parent / "images" / "heart-rate.png"  
st.set_page_config(
    page_title="Heart Disease Risk Predictor & Model Evaluation",
    page_icon=str(ICON_PATH),  # favicon
    layout="wide"
)


# ----------------------------------
# Constants
# ----------------------------------
DEFAULT_THRESHOLD = 0.08
TARGET_COL = "HeartDisease"

# Built-in logistic regression coefficients from statsmodels (post-VIF)
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

# Feature order - CRITICAL for model compatibility
EXPECTED_FEATURES = [
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
    """Safely load a pickle/joblib file"""
    if file_uploader is None:
        return None
    
    try:
        file_bytes = file_uploader.getvalue()
        
        # Try pickle first
        try:
            return pickle.load(io.BytesIO(file_bytes))
        except Exception:
            # Fallback to joblib
            return joblib.load(io.BytesIO(file_bytes))
            
    except Exception as e:
        st.error(f"❌ Failed to load {file_uploader.name}: {str(e)}")
        return None


def validate_features(df: pd.DataFrame, expected_features: list) -> Tuple[bool, str]:
    """
    Validate that dataframe has all expected features
    Returns: (is_valid, error_message)
    """
    df_features = set(df.columns)
    expected_set = set(expected_features)
    
    missing = expected_set - df_features
    extra = df_features - expected_set
    
    if missing:
        return False, f"Missing features: {sorted(missing)}"
    
    if extra:
        st.warning(f"⚠️ Extra features in data (will be ignored): {sorted(extra)}")
    
    return True, ""


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


def predict_single_case(X_input: dict, mode: str, model, encoder) -> Tuple[float, str, str]:
    """
    Make prediction for a single case
    Returns: (probability, risk_category, method_used)
    """
    df_input = pd.DataFrame([X_input], columns=EXPECTED_FEATURES)
    
    if mode == "Load a trained scikit‑learn model (.pkl)" and model is not None:
        try:
            X_use = df_input.copy()
            
            # Apply encoder if provided
            if encoder is not None:
                X_use = encoder.transform(X_use)
            else:
                # Ensure numeric for models that need it
                X_use = ensure_numeric(X_use)
            
            proba = model.predict_proba(X_use)[0, 1]
            return proba, categorize_risk(proba), "Uploaded Model"
            
        except Exception as e:
            st.warning(f"⚠️ Model prediction failed: {e}. Falling back to built-in coefficients.")
    
    # Built-in coefficient path
    z = BUILT_IN_COEFFICIENTS["const"]
    for feat in EXPECTED_FEATURES:
        z += BUILT_IN_COEFFICIENTS[feat] * X_input[feat]
    
    proba = sigmoid(z)
    return proba, categorize_risk(proba), "Built-in Coefficients"


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
if "suggest_threshold_clicked" not in st.session_state:
    st.session_state.suggest_threshold_clicked = False
if "eval_proba" not in st.session_state:
    st.session_state.eval_proba = None
if "eval_y_true" not in st.session_state:
    st.session_state.eval_y_true = None

# ----------------------------------
# Header
# ----------------------------------
APP_DIR = Path(__file__).resolve().parent
ICON_PATH = APP_DIR.parent / "images" / "heart-rate.png"

try:
    if ICON_PATH.exists():
        icon_img = Image.open(ICON_PATH)
        col1, col2 = st.columns([0.15, 3])
        with col1:
            st.image(icon_img, width=125)
        with col2:
            st.title("Heart Disease Predictor & Model Evaluation")
    else:
        st.title("Heart Disease Predictor & Model Evaluation")
except Exception:
    st.title("Heart Disease Predictor & Model Evaluation")

st.markdown("""
<div style='background-color: #1e3a5f; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <h4 style='color: #ffffff; margin: 0;'>📚 How This App Works</h4>
    <p style='color: #e0e0e0; margin: 10px 0 0 0;'>
        This app has <b>two main functions</b>:
    </p>
    <ol style='color: #e0e0e0; margin: 5px 0;'>
        <li><b>Single Case Prediction:</b> Predict heart disease risk for ONE patient using their health information</li>
        <li><b>Model Evaluation:</b> Test how well your model performs on a DATASET of many patients (requires CSV with features + 'HeartDisease' column)</li>
    </ol>
    <p style='color: #e0e0e0; margin: 5px 0 0 0;'>
        <b>💡 Tip:</b> For best results with uploaded models, ensure your CSV has the exact same feature preprocessing used during training.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------
# Sidebar: Model Selection
# ----------------------------------
st.sidebar.header("⚙️ Model Selection")

mode = st.sidebar.radio(
    "Choose Prediction Backend:",
    ["Built‑in Logistic Regression (coefficients)", "Load a trained scikit‑learn model (.pkl)"],
    index=0,
    help="Built-in: Uses coefficients from statsmodels. Upload: Use your trained model."
)

model = None
encoder = None
model_info = {}

if mode == "Load a trained scikit‑learn model (.pkl)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Upload Model Files")
    
    uploaded_model = st.sidebar.file_uploader(
        "Upload trained model (.pkl)", 
        type=["pkl"],
        key="model_uploader",
        help="Upload your trained Random Forest, Logistic Regression, or other sklearn model"
    )
    
    uploaded_encoder = st.sidebar.file_uploader(
        "(Optional) Upload encoder/transformer (.pkl)", 
        type=["pkl"],
        key="encoder_uploader",
        help="If you used preprocessing (StandardScaler, LabelEncoder, etc.), upload it here"
    )
    
    # Load the model
    if uploaded_model:
        model = load_pickle_file(uploaded_model)
        if model is not None:
            st.sidebar.success(f"✅ Model loaded: {uploaded_model.name}")
            
            # Try to extract model info
            try:
                model_name = type(model).__name__
                model_info["name"] = model_name
                st.sidebar.info(f"📊 Model type: {model_name}")
                
                # Check if model has feature_names_in_
                if hasattr(model, 'feature_names_in_'):
                    model_info["features"] = list(model.feature_names_in_)
                    st.sidebar.write(f"🔢 Expected features ({len(model.feature_names_in_)}): ")
                    with st.sidebar.expander("Show feature names"):
                        st.write(model.feature_names_in_)
            except Exception as e:
                st.sidebar.warning(f"Could not extract model info: {e}")
    
    # Load the encoder
    if uploaded_encoder:
        encoder = load_pickle_file(uploaded_encoder)
        if encoder is not None:
            st.sidebar.success(f"✅ Encoder loaded: {uploaded_encoder.name}")
            encoder_name = type(encoder).__name__
            st.sidebar.info(f"🔄 Encoder type: {encoder_name}")

else:
    st.sidebar.info("ℹ️ Using built-in logistic regression coefficients from your statsmodels output")

# ----------------------------------
# Sidebar: Threshold Control
# ----------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🎯 Classification Threshold")
st.sidebar.markdown(f"""
Current threshold: **{st.session_state.threshold:.3f}**

Predictions ≥ threshold → **Positive** (Heart Disease)  
Predictions < threshold → **Negative** (No Heart Disease)
""")

threshold = st.sidebar.slider(
    "Adjust Threshold:",
    min_value=0.05,
    max_value=0.95,
    value=st.session_state.threshold,
    step=0.01,
    key="threshold_slider",
    help="Higher threshold = fewer positive predictions (higher precision, lower recall)"
)
st.session_state.threshold = threshold

# ----------------------------------
# Tab Layout for Better Organization
# ----------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Single Case Prediction", "📊 Model Evaluation", "ℹ️ Help & Documentation"])

# ----------------------------------
# TAB 1: Single Case Prediction
# ----------------------------------
with tab1:
    st.header("📝 Patient Information")
    st.markdown("Enter patient details below to predict their heart disease risk.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        sex = st.selectbox("Sex", ["Female", "Male"])
        age_label_to_code = {
            "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4, "40–44": 5, "45–49": 6,
            "50–54": 7, "55–59": 8, "60–64": 9, "65–69": 10, "70–74": 11, "75–79": 12, "80+": 13
        }
        age_label = st.selectbox("Age Category", list(age_label_to_code.keys()), index=8)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=28.0, step=0.1)
        sleep_time = st.number_input("Average Sleep Time (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    
    with col2:
        st.subheader("Lifestyle Factors")
        smoking = st.selectbox("Currently Smokes?", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Drinking?", ["No", "Yes"])
        physical_activity = st.selectbox("Physical Activity?", ["No", "Yes"])
        diff_walking = st.selectbox("Difficulty Walking?", ["No", "Yes"])
    
    with col3:
        st.subheader("Health Conditions")
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
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            proba, risk_category, method = predict_single_case(X_input, mode, model, encoder)
        
        # Display results
        st.markdown("### 📊 Prediction Results")
        
        st.info(f"**Method used:** {method}")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Risk Probability", f"{proba:.1%}")
        with res_col2:
            st.metric("Risk Category", risk_category)
        with res_col3:
            classification = "HIGH RISK ⚠️" if proba >= threshold else "LOW RISK ✅"
            st.metric("Classification", classification)
        
        # Risk interpretation
        if proba < threshold:
            st.success(f"✅ **Low Risk**: Probability ({proba:.3f}) is below threshold ({threshold:.3f})")
        else:
            st.error(f"⚠️ **High Risk**: Probability ({proba:.3f}) is above threshold ({threshold:.3f})")
        
        # Visual probability bar
        fig_bar, ax_bar = plt.subplots(figsize=(10, 2), dpi=100)
        colors = ['green' if proba < threshold else 'red']
        ax_bar.barh([0], [proba], color=colors, alpha=0.7, height=0.5)
        ax_bar.axvline(threshold, color='black', linestyle='--', linewidth=2, 
                      label=f'Threshold ({threshold:.3f})')
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Probability of Heart Disease", fontsize=12)
        ax_bar.set_yticks([])
        ax_bar.legend(loc='upper right')
        ax_bar.grid(axis='x', alpha=0.3)
        st.pyplot(fig_bar)
        plt.close()
        
        # Risk factors summary
        with st.expander("📋 See Risk Factor Breakdown"):
            st.markdown("**High-Risk Factors Present:**")
            high_risk = []
            if stroke == "Yes":
                high_risk.append("• Stroke history (highest risk factor)")
            if kidney == "Yes":
                high_risk.append("• Kidney disease (very high risk)")
            if diabetic == "Yes":
                high_risk.append("• Diabetes")
            if smoking == "Yes":
                high_risk.append("• Smoking")
            if sex == "Male":
                high_risk.append("• Male gender")
            if diff_walking == "Yes":
                high_risk.append("• Difficulty walking")
            
            if high_risk:
                for factor in high_risk:
                    st.write(factor)
            else:
                st.write("• No major risk factors identified")
            
            st.markdown("**Protective Factors Present:**")
            protective = []
            if physical_activity == "Yes":
                protective.append("• Regular physical activity")
            if alcohol == "Yes":
                protective.append("• Moderate alcohol consumption")
            
            if protective:
                for factor in protective:
                    st.write(factor)
            else:
                st.write("• No protective factors identified")

# ----------------------------------
# TAB 2: Model Evaluation
# ----------------------------------
with tab2:
    st.header("📈 Model Performance Evaluation")
    st.markdown("""
    Upload a CSV file with patient data to evaluate your model's performance. 
    
    **Requirements:**
    - CSV must include all 15 feature columns
    - Must have a `HeartDisease` column (0 or 1)
    - Features should be preprocessed the same way as during training
    """)
    
    # Display current model info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            f"""<div style="padding:1rem; border:2px solid #4CAF50; border-radius:10px; 
            background-color:#1e3a5f; margin:10px 0;">
            <b style='color:#ffffff;'>🎯 Model in Use:</b> 
            <span style='color:#4CAF50; font-size:18px;'>{mode}</span>
            </div>""",
            unsafe_allow_html=True
        )
    
    with col2:
        if st.button("🔄 Reset Threshold to 0.08"):
            st.session_state.threshold = DEFAULT_THRESHOLD
            st.rerun()
    
    # File upload
    eval_file = st.file_uploader(
        "📄 Upload Evaluation CSV",
        type=["csv"],
        help="CSV with features and 'HeartDisease' target column",
        key="eval_file_uploader"
    )
    
    # Main evaluation logic
    if eval_file is not None:
        try:
            df = pd.read_csv(eval_file)
            st.success(f"✅ Loaded {len(df):,} records from {eval_file.name}")
            
            # Show data preview
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head(10))
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {list(df.columns)}")
            
            # Validate target column
            if TARGET_COL not in df.columns:
                st.error(f"❌ CSV must include the binary target column '{TARGET_COL}'.")
                st.stop()
            
            # Extract features and target
            y_true = df[TARGET_COL].astype(int).values
            X = df.drop(columns=[TARGET_COL])
            
            # Validate features
            is_valid, error_msg = validate_features(X, EXPECTED_FEATURES)
            if not is_valid:
                st.error(f"❌ Feature validation failed: {error_msg}")
                st.error(f"Expected features: {EXPECTED_FEATURES}")
                st.error(f"Found features: {list(X.columns)}")
                st.stop()
            
            # Reorder features to match expected order
            X = X[EXPECTED_FEATURES]
            
            st.info(f"📊 Target distribution: {(y_true==1).sum():,} positive ({(y_true==1).mean()*100:.2f}%), "
                   f"{(y_true==0).sum():,} negative ({(y_true==0).mean()*100:.2f}%)")
            
            backend_used = mode
            proba = None
            
            # Try uploaded model first if selected
            if mode.startswith("Load") and model is not None:
                try:
                    with st.spinner("🔄 Generating predictions with uploaded model..."):
                        X_use = X.copy()
                        
                        # Apply encoder if provided
                        if encoder is not None:
                            st.info("🔄 Applying encoder transformation...")
                            X_use = encoder.transform(X_use)
                        else:
                            st.info("ℹ️ No encoder provided. Converting to numeric codes...")
                            X_use = ensure_numeric(X_use)
                        
                        # Make predictions
                        proba = model.predict_proba(X_use)[:, 1]
                        
                        st.success(f"✅ Successfully generated predictions using: {uploaded_model.name}")
                        
                        # Show prediction distribution
                        fig_hist, ax_hist = plt.subplots(figsize=(8, 4), dpi=100)
                        ax_hist.hist(proba, bins=50, edgecolor='black', alpha=0.7)
                        ax_hist.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                                      label=f'Threshold ({threshold:.3f})')
                        ax_hist.set_xlabel("Predicted Probability")
                        ax_hist.set_ylabel("Frequency")
                        ax_hist.set_title("Distribution of Predicted Probabilities")
                        ax_hist.legend()
                        ax_hist.grid(alpha=0.3)
                        
                        with st.expander("📊 View Prediction Distribution"):
                            st.pyplot(fig_hist)
                        plt.close()
                        
                except Exception as e:
                    st.error(f"❌ Uploaded model failed: {str(e)}")
                    st.error("**Debugging Info:**")
                    st.write(f"- Model type: {type(model)}")
                    st.write(f"- Input shape: {X_use.shape}")
                    st.write(f"- Features: {list(X.columns)}")
                    
                    st.warning("⚠️ Falling back to built-in Logistic Regression...")
                    backend_used = "Built-in Logistic Regression (fallback)"
                    proba = None
            
            # Built-in LR path (with cross-validation)
            if proba is None:
                with st.spinner("🔄 Training built-in Logistic Regression with 5-fold CV..."):
                    X_enc = ensure_numeric(X)
                    
                    try:
                        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        blr = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
                        proba = cross_val_predict(blr, X_enc, y_true, cv=skf, method="predict_proba")[:, 1]
                        
                        st.success("✅ Built-in Logistic Regression trained successfully (5-fold CV)")
                        
                    except Exception as e:
                        st.error(f"❌ Built-in Logistic Regression failed: {e}")
                        st.stop()
            
            # Store in session state for threshold suggestion
            st.session_state.eval_proba = proba
            st.session_state.eval_y_true = y_true
            
            # Handle threshold suggestion
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✨ Suggest Best Threshold (Youden's J)", use_container_width=True):
                    try:
                        suggested_thr = compute_youden_threshold(y_true, proba)
                        st.session_state.threshold = round(suggested_thr, 3)
                        st.success(f"✅ Optimal threshold set to: {suggested_thr:.3f}")
                        st.info("🔄 Scroll down to see updated metrics. You can also adjust manually using the sidebar.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Could not compute Youden's J: {e}")
            
            with col_b:
                st.markdown(f"**Current threshold:** {st.session_state.threshold:.3f}")
            
            # Compute predictions and metrics
            thr = float(st.session_state.threshold)
            y_pred = (proba >= thr).astype(int)
            
            # --- Compute Metrics ---
            auc = roc_auc_score(y_true, proba)
            acc = accuracy_score(y_true, y_pred)
            sens = recall_score(y_true, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            # Display metrics
            st.markdown("---")
            st.markdown(f"### 📊 Performance Metrics (Threshold = {thr:.3f})")
            
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("🎯 AUC", f"{auc:.4f}")
            k2.metric("📈 Sensitivity", f"{sens:.4f}", help="True Positive Rate (Recall)")
            k3.metric("📉 Specificity", f"{spec:.4f}", help="True Negative Rate")
            k4.metric("✅ Accuracy", f"{acc:.4f}")
            k5.metric("🎪 Precision", f"{ppv:.4f}", help="Positive Predictive Value")
            
            # Additional metrics
            st.markdown("#### 📋 Detailed Metrics")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.write(f"**True Positives (TP):** {tp:,}")
                st.write(f"**False Positives (FP):** {fp:,}")
                st.write(f"**Positive Predictive Value (PPV):** {ppv:.4f}")
            with metric_col2:
                st.write(f"**True Negatives (TN):** {tn:,}")
                st.write(f"**False Negatives (FN):** {fn:,}")
                st.write(f"**Negative Predictive Value (NPV):** {npv:.4f}")
            
            # --- Visualizations ---
            st.markdown("---")
            
            # ROC Curve
            col_roc, col_cm = st.columns(2)
            
            with col_roc:
                st.markdown("### 📈 ROC Curve")
                fpr, tpr, thresholds_roc = roc_curve(y_true, proba)
                
                fig_roc, ax_roc = plt.subplots(figsize=(7, 6), dpi=120)
                ax_roc.plot(fpr, tpr, lw=2.5, label=f'Model (AUC = {auc:.4f})', color='#2E86AB')
                ax_roc.plot([0, 1], [0, 1], '--', lw=1.5, color='gray', label='Random')
                
                # Mark current threshold on ROC curve
                idx = np.argmin(np.abs(thresholds_roc - thr))
                ax_roc.plot(fpr[idx], tpr[idx], 'ro', markersize=10, 
                          label=f'Current Threshold ({thr:.3f})')
                
                ax_roc.set_xlabel("False Positive Rate", fontsize=11)
                ax_roc.set_ylabel("True Positive Rate", fontsize=11)
                ax_roc.set_title(f"ROC Curve", fontsize=13, fontweight='bold')
                ax_roc.legend(loc='lower right')
                ax_roc.grid(alpha=0.3)
                st.pyplot(fig_roc)
                plt.close()
            
            with col_cm:
                st.markdown("### 🔢 Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5), dpi=120)
                im = ax_cm.imshow(cm, cmap='Blues', alpha=0.8)
                
                ax_cm.set_xticks([0, 1])
                ax_cm.set_yticks([0, 1])
                ax_cm.set_xticklabels(['Predicted Neg', 'Predicted Pos'])
                ax_cm.set_yticklabels(['Actual Neg', 'Actual Pos'])
                
                # Add text annotations
                for i in range(2):
                    for j in range(2):
                        text = ax_cm.text(j, i, f'{cm[i, j]:,}',
                                        ha="center", va="center", color="black", 
                                        fontsize=16, fontweight='bold')
                
                ax_cm.set_title(f"Confusion Matrix (Threshold = {thr:.3f})", 
                              fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=ax_cm)
                st.pyplot(fig_cm)
                plt.close()
            
            # Calibration Curve
            st.markdown("### 📉 Calibration Curve")
            try:
                prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10, strategy='uniform')
                
                fig_cal, ax_cal = plt.subplots(figsize=(8, 6), dpi=120)
                ax_cal.plot(prob_pred, prob_true, marker='o', lw=2, markersize=8, 
                          color='#A23B72', label='Model Calibration')
                ax_cal.plot([0, 1], [0, 1], '--', lw=1.5, color='gray', label='Perfect Calibration')
                ax_cal.set_xlabel("Predicted Probability", fontsize=11)
                ax_cal.set_ylabel("Observed Frequency", fontsize=11)
                ax_cal.set_title("Calibration Plot", fontsize=13, fontweight='bold')
                ax_cal.legend(loc='upper left')
                ax_cal.grid(alpha=0.3)
                st.pyplot(fig_cal)
                plt.close()
            except Exception as e:
                st.warning(f"Could not generate calibration curve: {e}")
            
            # Performance summary
            st.markdown("---")
            st.markdown("### 📝 Performance Summary")
            st.markdown(f"""
            **Model:** {backend_used}  
            **AUC Score:** {auc:.4f} {'🟢' if auc > 0.8 else '🟡' if auc > 0.7 else '🔴'}  
            **Accuracy:** {acc:.4f}  
            **Sensitivity:** {sens:.4f} (Ability to identify positive cases)  
            **Specificity:** {spec:.4f} (Ability to identify negative cases)  
            **Precision:** {ppv:.4f} (Accuracy of positive predictions)
            
            {'✅ **Excellent performance!** AUC > 0.8 indicates strong discriminative ability.' if auc > 0.8 else 
             '⚠️ **Moderate performance.** Consider feature engineering or trying different models.' if auc > 0.7 else
             '❌ **Poor performance.** Model may need significant improvement.'}
            """)
            
        except Exception as e:
            st.error(f"❌ Error processing evaluation file: {str(e)}")
            st.exception(e)
    else:
        st.info("📁 Upload a CSV file above to begin model evaluation.")
        
        # Show example CSV format
        with st.expander("📋 See Example CSV Format"):
            example_data = pd.DataFrame({
                'BMI': [28.0, 32.5],
                'Sex': [1, 0],
                'AgeCategory': [9, 11],
                'Smoking': [1, 0],
                'AlcoholDrinking': [0, 1],
                'DiffWalking': [0, 1],
                'PhysicalActivity': [1, 0],
                'SleepTime': [7.0, 6.5],
                'PhysicalHealth': [3, 5],
                'MentalHealth': [3, 10],
                'Asthma': [0, 1],
                'KidneyDisease': [0, 0],
                'SkinCancer': [0, 0],
                'Stroke': [0, 1],
                'Diabetic': [1, 1],
                'HeartDisease': [1, 0]
            })
            st.dataframe(example_data)
            st.caption("Your CSV should have these 16 columns (15 features + HeartDisease target)")

# ----------------------------------
# TAB 3: Help & Documentation
# ----------------------------------
with tab3:
    st.header("ℹ️ Help & Documentation")
    
    st.markdown("""
    ## 🎯 Understanding the Two Main Functions
    
    ### 1. 🔮 Single Case Prediction
    **Purpose:** Predict heart disease risk for ONE individual patient
    
    **How it works:**
    - Enter a patient's demographic and health information
    - The model calculates a probability (0-1) of heart disease
    - If probability ≥ threshold → classified as HIGH RISK
    - If probability < threshold → classified as LOW RISK
    
    **Use cases:**
    - Clinical screening for individual patients
    - Risk assessment during medical consultations
    - Educational demonstrations
    
    ---
    
    ### 2. 📊 Model Evaluation
    **Purpose:** Assess how well your model performs on a DATASET of many patients
    
    **How it works:**
    - Upload a CSV with features + actual outcomes (HeartDisease column)
    - Model makes predictions for ALL patients
    - Compare predictions vs. actual outcomes
    - Calculate performance metrics (AUC, accuracy, sensitivity, etc.)
    
    **Use cases:**
    - Validating model performance on new data
    - Comparing different models
    - Finding optimal classification threshold
    - Research and model development
    
    ---
    
    ## 🔧 Model Types
    
    ### Built-in Logistic Regression (Coefficients)
    - Uses fixed coefficients from your statsmodels output
    - For single predictions: Direct calculation using sigmoid function
    - For evaluation: Trains a new sklearn LogisticRegression with 5-fold CV
    - **Pros:** Fast, interpretable, no upload needed
    - **Cons:** May not capture non-linear relationships
    
    ### Uploaded Model (.pkl)
    - Use your trained Random Forest, SVM, or other sklearn model
    - **CRITICAL:** Your CSV data must be preprocessed exactly as during training
    - If you used StandardScaler, LabelEncoder, etc., upload the transformer too
    - **Pros:** Can use more sophisticated models, better performance
    - **Cons:** Requires careful preprocessing alignment
    
    ---
    
    ## 🎚️ Understanding the Threshold
    
    The **threshold** determines the cutoff for classification:
    - **Lower threshold** (e.g., 0.05): More patients classified as high risk
      - ✅ Catches more true positives (higher sensitivity)
      - ❌ More false alarms (lower specificity)
      - **Use when:** Missing a positive case is very costly
    
    - **Higher threshold** (e.g., 0.50): Fewer patients classified as high risk
      - ✅ Fewer false alarms (higher specificity)
      - ❌ Misses more true positives (lower sensitivity)
      - **Use when:** False alarms are costly
    
    **Youden's J:** Finds the threshold that maximizes (Sensitivity + Specificity - 1)
    - Balances true positive and true negative rates
    - Your model found optimal threshold ≈ 0.078 during training
    
    ---
    
    ## 📊 Key Metrics Explained
    
    | Metric | What it Measures | Good Value |
    |--------|-----------------|------------|
    | **AUC** | Overall discriminative ability | > 0.8 |
    | **Accuracy** | % of correct predictions | > 0.85 |
    | **Sensitivity (Recall)** | % of actual positives correctly identified | > 0.80 |
    | **Specificity** | % of actual negatives correctly identified | > 0.80 |
    | **Precision (PPV)** | % of positive predictions that are correct | Varies |
    
    ---
    
    ## 🐛 Troubleshooting
    
    ### "Poor AUC with my uploaded model"
    **Likely causes:**
    1. CSV features not preprocessed the same way as training data
    2. Feature order doesn't match what model expects
    3. Encoding/scaling not applied
    
    **Solutions:**
    - Upload the encoder/transformer you used during training
    - Ensure CSV has exact same feature engineering as training
    - Check that feature names and order match
    
    ### "Threshold suggestion not working"
    - Ensure you've uploaded evaluation data first
    - Click "Suggest Best Threshold" AFTER model evaluation completes
    - Check that HeartDisease column has both 0 and 1 values
    
    ### "Model fails to load"
    - Ensure model was saved with pickle or joblib
    - Check Python version compatibility
    - Try re-saving model with: `joblib.dump(model, 'model.pkl')`
    
    ---
    
    ## 📚 Feature Descriptions
    
    | Feature | Type | Description |
    |---------|------|-------------|
    | BMI | Numeric | Body Mass Index |
    | Sex | Binary | 0=Female, 1=Male |
    | AgeCategory | Ordinal | 1-13 (18-24 to 80+) |
    | Smoking | Binary | Currently smokes |
    | AlcoholDrinking | Binary | Heavy drinking |
    | DiffWalking | Binary | Difficulty walking |
    | PhysicalActivity | Binary | Regular exercise |
    | SleepTime | Numeric | Hours per night |
    | PhysicalHealth | Numeric | Poor health days (0-30) |
    | MentalHealth | Numeric | Poor mental health days (0-30) |
    | Asthma | Binary | Has asthma |
    | KidneyDisease | Binary | Has kidney disease |
    | SkinCancer | Binary | Has skin cancer |
    | Stroke | Binary | Prior stroke |
    | Diabetic | Binary | Has diabetes |
    
    ---
    
    ## 💡 Best Practices
    
    1. **For Single Predictions:**
       - Verify patient data accuracy before predicting
       - Consider multiple risk factors together
       - Use clinical judgment alongside model predictions
    
    2. **For Model Evaluation:**
       - Use held-out test data (not training data)
       - Ensure sufficient sample size (> 1000 records)
       - Check class balance in your evaluation data
       - Compare multiple thresholds
    
    3. **Model Deployment:**
       - Document your preprocessing steps
       - Save encoder/scaler with your model
       - Test on diverse patient populations
       - Regularly update with new data
    """)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.markdown("#### ⚠️ Disclaimer")
st.markdown("*This application is for educational and research purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult healthcare professionals for medical decisions.*")

st.markdown("---")
st.markdown("### 📬 Contact")
st.markdown("**William C. Phiri** | [GitHub](https://github.com/kochezz) | wphiri@beda.ie")
st.markdown("© 2025 William C. Phiri – Powered by BEDA")
