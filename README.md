# ❤️ Heart Disease Prediction using Binary Logistic Regression (BLR)

[![Built With Python](https://img.shields.io/badge/Built%20With-Python-blue?logo=python)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data](https://img.shields.io/badge/Data-Cleaned-lightgrey)]()

---

## 📘 Project Overview

This project develops a **Binary Logistic Regression (BLR)** model to predict the likelihood of **heart disease** using health and lifestyle indicators.  
The workflow was executed in **VS Code** through three structured notebooks — Data Management, EDA, and Modeling — forming an end-to-end data science pipeline.

In addition to the logistic regression baseline, **three other models** (Naïve Bayes, SVM, and Random Forest) were trained and compared to identify the best-performing approach.  
The final model achieved a **Test AUC of 0.8269** and **Accuracy of 91.56%**, with **Random Forest** selected for deployment due to its balanced performance and generalization.

---

## 🎯 Objectives

- Clean, prepare, and validate a large health dataset for modeling.
- Investigate relationships between demographic and behavioral variables and heart disease risk.
- Build and interpret a **logistic regression model** to quantify associations between predictors and disease risk.
- Assess **multicollinearity** using VIF and iteratively remove redundant variables.
- Compare logistic regression with **Naïve Bayes**, **SVM (Linear)**, and **Random Forest** classifiers.
- Identify the **best-performing model** based on accuracy, AUC, precision, recall, and overfitting score.

---

## 📁 Project Structure

```
CAPSTONE_PROJECT/
├── dashboards/              # (Optional) Streamlit or visualization components
├── data/                    # Raw and processed datasets
├── environment/             # Environment configuration (requirements)
├── models/                  # Trained models and metadata (.pkl, .json)
├── notebooks/               # Jupyter notebooks
│   ├── 01_Data_Management.ipynb
│   ├── 02_EDA.ipynb
│   └── 03_Modeling.ipynb
├── reports/                 # Model comparison results and visual reports
├── src/                     # Scripts for model training and evaluation
├── initial_model_results.csv # Baseline model summary
├── main.py                  # Entrypoint for execution
└── README.md
```

---

## 🧩 Workflow Summary

| Notebook | Description |
|-----------|-------------|
| **01_Data_Management** | Data import, cleaning, missing value handling (e.g., diabetic variable imputation), and preparation of 15 predictive features. |
| **02_EDA** | Exploratory analysis showing categorical and numeric risk patterns — e.g., stroke (4–5× risk), diabetes (3×), and difficulty walking (3–4×). Detected class imbalance (8.56% positive cases). |
| **03_Modeling** | Model training, evaluation, and comparison. Implemented VIF-based feature reduction, logistic regression interpretation, odds ratios, and performance benchmarking across models. |

---

## 🧠 Model Development Summary

| Step | Description |
|------|-------------|
| **Model Type** | Binary Logistic Regression (Baseline) |
| **Target Variable** | HeartDisease (1 = Yes, 0 = No) |
| **Predictors (15)** | BMI, AgeCategory, Sex, Smoking, AlcoholDrinking, PhysicalActivity, SleepTime, PhysicalHealth, MentalHealth, DiffWalking, Stroke, Asthma, KidneyDisease, SkinCancer, Diabetic |
| **Class Imbalance** | 8.56% Heart Disease, 91.44% No Heart Disease |
| **Feature Reduction** | Removed high-VIF features (`SleepTime` and `BMI`) to improve model stability |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-score, Specificity, ROC-AUC |

---

## 📊 Model Results

### ⚙ Logistic Regression Performance
| Metric | Score |
|--------|--------|
| Accuracy | 91.49% |
| Sensitivity (Recall) | 9.12% |
| Specificity | 99.20% |
| Precision | 51.5% |
| F1-Score | 0.156 |
| AUC | 0.8266 |

**Significant Predictors (p < 0.05):**
- ↑ Risk: **Stroke**, **Sex (Male)**, **KidneyDisease**, **Diabetic**, **Smoking**, **DiffWalking**, **Asthma**, **AgeCategory**
- ↓ Risk: **AlcoholDrinking**, **PhysicalActivity**

**Interpretation:**  
- People with **stroke history** have **3.14× higher odds** of heart disease.  
- **Males** are roughly **2.1× more likely** to develop heart disease.  
- **Alcohol consumption** and **physical activity** show protective effects.

---

## 📊 Model Performance Comparison

| Model | Train_Accuracy | Test_Accuracy | Train_AUC | Test_AUC | Test_Sensitivity | Test_Precision | Test_F1 | Train_Time_sec | Overfit_Score |
|-------|----------------|---------------|-----------|----------|------------------|----------------|---------|----------------|---------------|
| Random Forest | 0.922736 | 0.915682 | 0.859329 | 0.826935 | 0.042192 | 0.600000 | 0.078840 | 9.34 | 0.032394 |
| Logistic Regression | 0.742132 | 0.741428 | 0.829478 | 0.826647 | 0.765114 | 0.215472 | 0.336250 | 0.32 | 0.002831 |
| Naive Bayes | 0.846277 | 0.845886 | 0.800363 | 0.797700 | 0.452237 | 0.265267 | 0.334391 | 0.08 | 0.002664 |
| SVM (Linear) | 0.301408 | 0.303929 | 0.556629 | 0.558903 | 0.771324 | 0.088922 | 0.159461 | 46.70 | -0.002274 |

---

### 🏆 BEST MODEL: Random Forest
- **Test AUC:** 0.8269
- **Test Accuracy:** 0.9156
- **Overfitting:** 0.0324

## 🏆 Final Model Selection: **Random Forest**

**Rationale:**
- Highest Test AUC (0.8269)
- Strong generalization (Overfit = 0.0324)
- Balanced across accuracy, precision, and recall  
- Capable of handling **non-linear relationships**  

**Saved Deliverables:**
- `best_model.pkl` — final Random Forest model  
- `model_metadata.json` — model parameters and performance summary  
- `Model_Comparison_Results.csv` — detailed benchmarking  
- `train_performance.csv` — performance summary  
- Visual comparisons (ROC, AUC, Radar, Overfitting Analysis) in `/reports`

---

## 🧮 Key Insights

- Stroke and kidney disease are the **strongest predictors** of heart disease.  
- Male gender, smoking, and diabetes **significantly increase risk**.  
- Regular physical activity and moderate alcohol consumption are **protective factors**.  
- Despite class imbalance (8.6%), model achieved **excellent specificity and AUC**.  
- Random Forest outperformed logistic regression, indicating **non-linear relationships** in the data.

---

## 🚀 Getting Started

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/kochezz/heart-disease-prediction-BLR.git
cd heart-disease-prediction-BLR

# Create environment
conda create -n heart-env python=3.10
conda activate heart-env

# Install dependencies
pip install -r environment/requirements.txt
```

### ▶️ Run the Pipeline

Open and execute the notebooks in sequence:

1. `01_Data_Management.ipynb`
2. `02_EDA.ipynb`
3. `03_Modeling.ipynb`

Or run the automated pipeline:
```bash
python main.py
```

---

## 🛠️ Tech Stack

- **Python 3.10**  
- pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn  
- joblib (for model persistence)  
- Jupyter / VS Code  

---

## 📬 Author

**William C. Phiri**  
[GitHub: @kochezz](https://github.com/kochezz)  
[LinkedIn](https://www.linkedin.com/in/william-phiri-866b8443/)  
📧 [wphiri@beda.ie](mailto:wphiri@beda.ie)

> 🧭 _“Get it done the BEDA way.”_

---

## 📄 License

This project is licensed under the **MIT License**.
