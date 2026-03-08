# 🧪 Toxicity Prediction using Machine Learning

A machine learning pipeline that predicts whether a chemical compound is **Toxic** or **NonToxic** based on molecular descriptors. The project covers the full data science workflow — from exploratory data analysis all the way through to cross-validated model comparison and final evaluation.

---

## 📁 Project Structure

```
toxicity-prediction/
│
├── toxicity_analysis.ipynb   # Main Jupyter Notebook with all code
├── data.csv                  # Dataset (171 compounds, 1203 molecular features)
└── README.md                 # Project documentation
```

---

## 📌 Problem Statement

Given a dataset of chemical compounds described by over 1,200 molecular descriptors, the goal is to build a binary classification model that can accurately predict whether a compound is:

- **Toxic** (1)
- **NonToxic** (0)

This kind of predictive model has real-world applications in drug discovery, environmental safety, and chemical risk assessment — helping researchers flag potentially harmful compounds early without expensive lab testing.

---

## 📊 Dataset Overview

| Property | Detail |
|----------|--------|
| Total samples | 171 compounds |
| Total features | 1,203 molecular descriptors |
| Target column | `Class` (Toxic / NonToxic) |
| Missing values | None |
| Class distribution | 115 NonToxic (67%) · 56 Toxic (33%) |

The dataset is **imbalanced** — there are roughly twice as many NonToxic compounds as Toxic ones, which is an important consideration during model evaluation.

---

## 🔬 Pipeline Overview

The project is broken into 6 steps, each in its own notebook cell:

### Step 1 — Data Cleaning
- Checked dataset shape, data types, and missing values
- Removed zero-variance features (no information content)
- Removed near-zero-variance features (threshold < 0.01)
- Removed highly correlated features (|r| > 0.95, keeping one from each pair)

| Stage | Features Remaining |
|-------|--------------------|
| Raw | 1,203 |
| After near-zero variance removal | 994 |
| After high-correlation removal | 560 |

### Step 2 — Preprocessing
- Encoded target column: `Toxic = 1`, `NonToxic = 0`
- Separated features (X) and target (y)
- Applied **StandardScaler** to normalise all features to mean=0, std=1

### Step 3 — Exploratory Data Analysis (EDA)
- Class distribution bar chart — confirmed class imbalance
- Feature variance distribution — showed most features have very low variance
- Correlation heatmap of the top 15 highest-variance features
- Boxplots comparing Toxic vs NonToxic distributions across top features

### Step 4 — Feature Selection
Used a **consensus voting approach** across 4 methods:

| Method | Features Selected |
|--------|------------------|
| ANOVA F-test | Top 30 |
| Mutual Information | Top 30 |
| Random Forest Importance | Top 30 |
| Extra Trees Importance | Top 30 |

Features appearing in the top 30 of **at least 2 methods** were kept, then the final **top 20** were selected by Random Forest importance score. This reduced the feature space from 560 down to just 20 highly informative features.

**Top selected features include:** `MDEC-23`, `AATSC3v`, `AATSC8i`, `SpAD_Dt`, `ATSC1v`, `ATSC3p`, `ATSC8i`, `MATS8i`, `SpMin3_Bhe`, `ATSC7p`

### Step 5 — Model Comparison with Cross-Validation
Four models were compared using **5-fold Stratified Cross-Validation**:

| Model | Accuracy | AUC-ROC | F1 |
|-------|----------|---------|-----|
| Random Forest | 0.672 ± 0.051 | **0.683 ± 0.077** | 0.339 |
| Gradient Boosting | 0.649 ± 0.054 | 0.619 ± 0.062 | 0.384 |
| Logistic Regression | 0.626 ± 0.053 | 0.588 ± 0.087 | 0.292 |
| SVM (RBF) | 0.708 ± 0.031 | 0.624 ± 0.111 | 0.297 |

**Random Forest** achieved the best AUC-ROC score of **0.683**, making it the selected model for final evaluation.

### Step 6 — Final Evaluation
The best model (Random Forest) was evaluated using:
- Out-of-fold predictions via `cross_val_predict`
- Full classification report (Precision, Recall, F1 per class)
- Confusion matrix heatmap
- ROC curve with AUC score

Full results are available in the notebook.

---

## 🧠 Key Findings

- The dataset is **high-dimensional** (1,203 features for only 171 samples), making feature selection critical to avoid overfitting
- **Feature selection reduced the problem** from 560 to just 20 features without sacrificing model performance
- **Random Forest** was the most consistent model across all metrics
- The moderate AUC-ROC (~0.68) is expected given the small dataset size and class imbalance — more data would likely improve performance significantly
- The **class imbalance** (67% NonToxic vs 33% Toxic) means accuracy alone is a misleading metric; AUC-ROC and F1 are more reliable indicators here

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3 | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-learn | ML models, feature selection, cross-validation |
| Matplotlib | Data visualisation |
| Seaborn | Statistical plots |
| Jupyter Notebook | Interactive development environment |
| Anaconda | Python environment management |

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/toxicity-prediction.git
   ```

2. Make sure you have the required libraries installed. If using Anaconda, open **Anaconda Navigator** and launch **Jupyter Notebook**

3. Open `toxicity_analysis.ipynb` in Jupyter

4. Make sure `data.csv` is in the same folder as the notebook

5. Run the cells **one by one in order** using **Shift + Enter**

---

## 📈 Possible Improvements

- Apply **SMOTE** or other oversampling techniques to address class imbalance
- Tune hyperparameters using **GridSearchCV** or **RandomizedSearchCV**
- Try additional models like **XGBoost** or **LightGBM**
- Use **SHAP values** to better interpret which molecular descriptors drive toxicity predictions
- Collect more data — 171 samples is small for a 1,200-feature problem

---

---


