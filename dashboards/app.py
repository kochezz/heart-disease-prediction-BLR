
# app.py — Regenerated (Capstone Heart Risk App)
# Notes:
# - Clear “Model in use” banner (persists via session_state)
# - Robust loader for uploaded model + optional encoder/transformer
# - Evaluation tab: ROC, PR, Calibration, Confusion Matrix + extra metrics
# - Threshold helpers: Reset, Youden’s J, Target Sensitivity 0.90
# - Dataset summary expander for quick sanity checks
# - Single prediction from a one-row CSV (schema-safe)
# - Patient next-steps guidance (educational, non-clinical)
# - History + downloads (results and evaluation report)
# - About tab with a lightweight Model Card
#
# How “Built‑in BLR” works in this version:
#   We TRAIN a LogisticRegression on the uploaded evaluation CSV using all columns
#   except the target (default "HeartDisease"). For ROC/PR, we use cross_val_predict
#   to generate out-of-fold probabilities to avoid training/test leakage.
#
# If you use a pre-trained model, upload a Pipeline (preferred) or a model + encoder.

import io
import os
import json
import pickle
import math
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score, recall_score,
    precision_score, precision_recall_curve, average_precision_score,
    f1_score, balanced_accuracy_score, matthews_corrcoef, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from io import BytesIO

st.set_page_config(
    page_title="Model Evaluation (ROC & Calibration)",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Utilities ---------------------------

def _load_pickle(file):
    try:
        return pickle.load(file)
    except Exception:
        file.seek(0)
        import joblib
        return joblib.load(file)

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce non-numeric columns with simple label encoding for demo purposes.
       If an encoder/pipeline is provided, prefer that instead.
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype("category").cat.codes
    return out

def compute_youden_threshold(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    best_idx = np.argmax(j)
    return max(min(thr[best_idx], 0.999), 0.001)

def pick_target_col(df, default="HeartDisease"):
    if default in df.columns:
        return default
    # fallback: last column if binary
    for c in df.columns[::-1]:
        vals = df[c].dropna().unique()
        if len(vals) <= 2 and set(vals).issubset({0,1}):
            return c
    return default

def b64_download(data: bytes, filename: str, mime: str="application/octet-stream"):
    import base64
    b64 = base64.b64encode(data).decode()
    href = f'<a download="{filename}" href="data:{mime};base64,{b64}">Download {filename}</a>'
    return href

def plot_roc(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0,1],[0,1],'--', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC = {auc:.3f})")
    ax.grid(alpha=.2)
    st.pyplot(fig)
    return fig, auc

def plot_pr(y_true, probs):
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(rec, prec, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall (AP = {ap:.3f})")
    ax.grid(alpha=.2)
    st.pyplot(fig)
    return fig, ap

def plot_calibration(y_true, probs, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='uniform')
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(prob_pred, prob_true, marker='o', lw=1.5)
    ax.plot([0,1],[0,1],'--', lw=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.grid(alpha=.2)
    st.pyplot(fig)
    return fig

def plot_confmat(cm, labels=("Neg","Pos")):
    fig, ax = plt.subplots(figsize=(5,4), dpi=150)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    ax.set_xticks([0,1], labels=["Pred 0","Pred 1"])
    ax.set_yticks([0,1], labels=["True 0","True 1"])
    # annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    st.pyplot(fig)
    return fig

def patient_guidance(prob, bands=(0.07, 0.15)):
    if prob < bands[0]:
        tier = "Low"
        recs = [
            "Maintain a balanced diet rich in fruits, vegetables, and whole grains.",
            "Aim for ≥150 minutes/week of moderate physical activity.",
            "Avoid tobacco exposure; moderate alcohol use if applicable.",
            "Keep routine check-ups and monitor blood pressure, glucose, and lipids."
        ]
    elif prob < bands[1]:
        tier = "Moderate"
        recs = [
            "Book a non-urgent appointment with your GP to review cardiovascular risk.",
            "Discuss screening labs (lipid panel, fasting glucose/HbA1c) and blood pressure monitoring.",
            "Adopt a structured activity plan and nutrition goals; consider a referral to lifestyle services.",
            "Review family history and other risk factors (sleep apnea, medications)."
        ]
    else:
        tier = "High"
        recs = [
            "Schedule a timely medical review to assess cardiovascular risk comprehensively.",
            "Discuss additional testing as appropriate (e.g., lipid panel, glucose/HbA1c, ECG).",
            "Create a supervised plan for risk factor control (BP, diabetes, smoking cessation).",
            "Seek urgent care for chest pain, shortness of breath at rest, or concerning symptoms."
        ]
    return tier, recs

# ---------------------- Session defaults ------------------------
if "backend_label" not in st.session_state:
    st.session_state.backend_label = "Built-in Logistic Regression (trained on CSV)"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------- Sidebar controls ------------------------
st.sidebar.header("Prediction Backend")
mode = st.sidebar.radio(
    "Choose a model source",
    ["Built-in Logistic Regression (trained on CSV)", "Load a trained model (.pkl)"],
)

st.session_state.backend_label = mode

uploaded_model = None
uploaded_encoder = None
loaded_model = None
loaded_encoder = None

if mode.startswith("Load"):
    uploaded_model = st.sidebar.file_uploader("Upload trained model (.pkl)", type=["pkl"])
    uploaded_encoder = st.sidebar.file_uploader("(Optional) Upload encoder/transformer (.pkl)", type=["pkl"])
    if uploaded_model:
        try:
            loaded_model = _load_pickle(uploaded_model)
            st.sidebar.success(f"Model loaded: {uploaded_model.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
    if uploaded_encoder:
        try:
            loaded_encoder = _load_pickle(uploaded_encoder)
            st.sidebar.info(f"Transformer loaded: {uploaded_encoder.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load transformer: {e}")

st.sidebar.header("Decision Threshold")
st.sidebar.slider("Threshold", 0.0, 1.0, key="threshold", value=st.session_state.threshold, step=0.01)

# ----------------------- Header & banner ------------------------
st.title("Model Evaluation (ROC & Calibration)")

st.markdown(
    f"""<div style="padding:.6rem 1rem;border:1px solid #444;border-radius:10px;display:inline-block;margin-bottom:.5rem;">
    <b>Model in use:</b> {st.session_state.backend_label}
    </div>""",
    unsafe_allow_html=True
)

# --------------------------- Tabs -------------------------------
tab_pred, tab_eval, tab_about = st.tabs(["🔮 Predict", "📈 Evaluate", "ℹ️ About"])

# ---------------------- Predict (single) ------------------------
with tab_pred:
    st.subheader("Single Prediction")
    st.caption("Upload a one-row CSV with the same feature columns used by your model (no target column).")

    one_row = st.file_uploader("Upload one-row CSV", type=["csv"], key="one_row_csv")
    if st.button("Predict", use_container_width=True, type="primary"):
        if one_row is None:
            st.error("Please upload a one-row CSV with feature columns.")
        else:
            df1 = pd.read_csv(one_row)
            if len(df1) != 1:
                st.error("Your CSV must contain exactly one row.")
            else:
                X = df1.copy()
                if loaded_encoder is not None:
                    try:
                        X = loaded_encoder.transform(X)
                    except Exception as e:
                        st.warning(f"Encoder transform failed; attempting numeric coercion: {e}")
                        X = ensure_numeric(X)
                else:
                    X = ensure_numeric(X)

                model = loaded_model
                backend_used = st.session_state.backend_label
                if model is None and mode.startswith("Load"):
                    st.error("Uploaded model could not be used; falling back to built-in BLR is not supported for single prediction without training data.")
                else:
                    if model is None:
                        st.error("For single prediction with the built-in BLR you must evaluate/train on a CSV in the Evaluate tab first.")
                    else:
                        # Predict
                        try:
                            proba = float(model.predict_proba(X)[0,1])
                        except Exception as e:
                            st.error(f"Model failed to predict: {e}")
                            proba = None

                        if proba is not None:
                            thr = float(st.session_state.threshold)
                            pred = int(proba >= thr)
                            st.metric("Predicted Probability", f"{proba:.3f}")
                            st.metric("Class @ Threshold", f"{pred} (threshold={thr:.2f})")
                            tier, recs = patient_guidance(proba, (0.07, 0.15))
                            st.markdown(f"### Suggested Next Steps ({tier} risk)")
                            for r in recs:
                                st.write("• " + r)
                            st.caption("Educational only; not medical advice. Please consult a healthcare professional.")
                            # history
                            item = {"probability": proba, "pred": pred, "threshold": thr}
                            item.update({f"feat:{k}": (float(v) if np.isscalar(v) and not isinstance(v, (str,bool)) else str(v)) for k,v in df1.iloc[0].items()})
                            st.session_state.history.append(item)

    if st.session_state.history:
        st.divider()
        st.subheader("Recent Predictions")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df.tail(10), use_container_width=True)
        csv_bytes = hist_df.to_csv(index=False).encode()
        st.download_button("Download History CSV", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv")

# ---------------------- Evaluate (CSV) --------------------------
with tab_eval:
    st.subheader("Evaluate a Dataset")
    eval_file = st.file_uploader("Upload CSV for evaluation (must include binary target column, default 'HeartDisease')", type=["csv"])
    pos_label = st.selectbox("Positive class label (for target column)", options=[1, "1", "true", "True"], index=0)

    col_thr1, col_thr2, col_thr3 = st.columns([1,1,1])
    with col_thr1:
        if st.button("Suggest Best Threshold (Youden’s J)") and eval_file is not None:
            # Will compute after proba_eval created
            st.session_state._suggest_j = True
    with col_thr2:
        if st.button("Target Sensitivity 0.90"):
            st.session_state._target_sens = 0.90
    with col_thr3:
        if st.button("Reset to 0.50"):
            st.session_state.threshold = 0.50

    if eval_file is not None:
        eval_df = pd.read_csv(eval_file)
        target_col = pick_target_col(eval_df, default="HeartDisease")
        if target_col not in eval_df.columns:
            st.error("No valid target column found. Include 'HeartDisease' or another binary column.")
        else:
            y_true_raw = eval_df[target_col]
            if isinstance(pos_label, str):
                y_true = (y_true_raw.astype(str) == str(pos_label)).astype(int).values
            else:
                y_true = (y_true_raw == pos_label).astype(int).values
            X = eval_df.drop(columns=[target_col])

            with st.expander("Dataset summary", expanded=False):
                st.write(f"Records: {len(eval_df):,}")
                st.write(f"Prevalence ({target_col}=Positive): {y_true.mean():.2%}")
                st.dataframe(eval_df.describe(include='all').transpose())

            # Choose model
            backend_used = st.session_state.backend_label
            model = loaded_model if loaded_model is not None else None

            proba_eval = None
            trained_fresh_model = None

            if backend_used.startswith("Load"):
                if model is None:
                    st.error("Uploaded model unavailable; using built-in BLR instead.")
                    backend_used = "Built-in Logistic Regression (trained on CSV)"
                else:
                    # Use uploaded model (prefer pipeline)
                    try:
                        X_use = X.copy()
                        if loaded_encoder is not None:
                            X_use = loaded_encoder.transform(X_use)
                        proba_eval = model.predict_proba(X_use)[:,1]
                    except Exception as e:
                        st.error(f"Uploaded model failed to predict ({e}); falling back to built-in BLR.")
                        backend_used = "Built-in Logistic Regression (trained on CSV)"
                        proba_eval = None

            if backend_used.startswith("Built-in") or proba_eval is None:
                # Train BLR with OOF probabilities for fair ROC/PR
                X_enc = ensure_numeric(X)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                blr = LogisticRegression(max_iter=200, solver="lbfgs")
                try:
                    proba_eval = cross_val_predict(blr, X_enc, y_true, cv=skf, method="predict_proba")[:,1]
                except Exception as e:
                    st.error(f"BLR training failed: {e}")
                    proba_eval = np.zeros_like(y_true, dtype=float)
                # Also fit on full data for downstream single predictions (optional)
                try:
                    blr.fit(X_enc, y_true)
                    trained_fresh_model = blr
                except Exception:
                    trained_fresh_model = None

            st.caption(f"Evaluation computed with: **{backend_used}**")

            # Threshold helpers that depend on proba_eval
            if "_suggest_j" in st.session_state and st.session_state._suggest_j:
                try:
                    st.session_state.threshold = round(float(compute_youden_threshold(y_true, proba_eval)), 3)
                    st.success(f"Set threshold to Youden’s J: {st.session_state.threshold:.3f}")
                except Exception as e:
                    st.warning(f"Could not compute Youden’s J: {e}")
                st.session_state._suggest_j = False

            if "_target_sens" in st.session_state and st.session_state._target_sens:
                target_sens = st.session_state._target_sens
                fpr, tpr, thr = roc_curve(y_true, proba_eval)
                idx = np.argmin(np.abs(tpr - target_sens))
                st.session_state.threshold = float(min(max(thr[idx], 0.001), 0.999))
                st.success(f"Set threshold for target sensitivity {target_sens:.2f}: {st.session_state.threshold:.3f}")
                st.session_state._target_sens = None

            thr = float(st.session_state.threshold)
            y_pred = (proba_eval >= thr).astype(int)

            # Metrics
            auc = roc_auc_score(y_true, proba_eval)
            acc = accuracy_score(y_true, y_pred)
            sens = recall_score(y_true, y_pred, zero_division=0)  # TPR
            # specificity = TN / (TN + FP)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            st.markdown("### Evaluation Summary")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("AUC", f"{auc:.3f}")
            k2.metric("Sensitivity (TPR)", f"{sens:.3f}")
            k3.metric("Specificity (TNR)", f"{spec:.3f}")
            k4.metric("Accuracy", f"{acc:.3f}")

            # Extra metrics
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            brier = brier_score_loss(y_true, proba_eval)

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Balanced Acc.", f"{bal_acc:.3f}")
            e2.metric("F1", f"{f1:.3f}")
            e3.metric("MCC", f"{mcc:.3f}")
            e4.metric("Brier Score", f"{brier:.3f}")

            # Plots
            st.markdown("### Curves")
            c1, c2 = st.columns(2)
            with c1:
                fig_roc, auc_val = plot_roc(y_true, proba_eval)
            with c2:
                fig_pr, ap_val = plot_pr(y_true, proba_eval)

            st.markdown("### Calibration & Confusion Matrix")
            c3, c4 = st.columns(2)
            with c3:
                fig_cal = plot_calibration(y_true, proba_eval)
            with c4:
                cm = confusion_matrix(y_true, y_pred)
                fig_cm = plot_confmat(cm)

            # Downloads: metrics + figures
            report_df = pd.DataFrame([{
                "AUC": auc, "Accuracy": acc, "Sensitivity": sens, "Specificity": spec,
                "BalancedAccuracy": bal_acc, "F1": f1, "MCC": mcc, "Brier": brier,
                "Threshold": thr, "Backend": backend_used
            }])
            st.download_button(
                "Download Metrics CSV",
                data=report_df.to_csv(index=False).encode(),
                file_name="evaluation_metrics.csv",
                mime="text/csv"
            )

            # Save figures to a zip in-memory
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for name, fig in [("roc.png", fig_roc), ("pr.png", fig_pr), ("calibration.png", fig_cal), ("confusion_matrix.png", fig_cm)]:
                    img = BytesIO()
                    fig.savefig(img, format="png", dpi=150, bbox_inches="tight")
                    zf.writestr(name, img.getvalue())
                zf.writestr("metrics.csv", report_df.to_csv(index=False).encode())
            st.download_button("Download Figures + Metrics (ZIP)", data=zip_buf.getvalue(), file_name="evaluation_report.zip", mime="application/zip")

            # Expose trained BLR for single predictions in this session
            if trained_fresh_model is not None and loaded_model is None:
                st.info("A built-in Logistic Regression was trained on this dataset for this session; you can now use it in the Predict tab with the same feature schema.")
                st.session_state._trained_blr = trained_fresh_model

# ----------------------------- About ----------------------------
with tab_about:
    st.subheader("Model Card (Lite)")
    st.markdown("""
**Intended use:** Educational demonstration of binary risk modeling and evaluation.  
**Not medical advice:** Outputs are for learning only and must not guide care without a qualified clinician.  
**Inputs:** Determined by the uploaded dataset and/or the uploaded model’s expected schema.  
**Backends:**  
- *Built-in Logistic Regression*: Trained on the evaluation CSV using numeric-encoded features.  
- *Uploaded model*: Prefer a scikit-learn `Pipeline` that bundles preprocessing and the estimator.  
**Evaluation:** ROC–AUC, Precision–Recall (AP), calibration plot, confusion matrix, and multiple summary metrics.  
**Thresholding:** Adjust with slider or choose suggested thresholds (Youden’s J, target sensitivity).  
**Limitations:** Domain shift, missing/incorrect encodings, small datasets, and label polarity can degrade performance.
""")

    st.divider()
    st.subheader("Contact")
    st.caption("Questions or feedback? Add a simple contact form here if needed.")

# ------------------------- Footer note --------------------------
st.caption("© Your Name — Capstone Project Demo | Built with Streamlit")
