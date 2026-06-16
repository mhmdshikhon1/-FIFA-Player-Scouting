# ⚽ FIFA Player Scouting — ML Assignments 2 & 3

> A full machine-learning pipeline built on a FIFA player dataset to **predict market value** (regression) and **classify performance tiers** (Low / Mid / High / Elite). Developed across two assignments by a 5-member team as part of the Machine Learning course.

---

## 👥 Team

| Member |
|---|
| Tarek Farag Abdelmobdy  |
| Mohamed Mahmoud Shaikhoun |
| Youssef Mohamed Abdel Fattah |
| Youssef Mohamed Ahmed |
| Abdelrahman Mohamed Mousa |

---

## 📂 Dataset

**FIFA Player Dataset (`Fifa.csv`)**  
Features include: `Age`, `Overall_Rating`, `Future Potential`, `Total_Stats Score`, `Position`, `Country`, `Team`, `Value Per M$`.  
Target variables:
- **Regression:** `Value Per M$` (player market value in millions)
- **Classification:** `Performance_Tier` — derived label with 4 classes: `Low`, `Mid`, `High`, `Elite`

---

## 🗂️ Project Structure

```
├── Assigment_3_ML__F.ipynb   # Main notebook (Assignments 2 + 3)
├── Fifa.csv                  # Dataset (place beside the notebook)
├── results.json              # Exported hyperparameters & CV scores
└── README.md
```

---

## 📋 Assignment 2 — Tasks Overview

### Task 1 — Exploratory Data Analysis
- Distribution analysis of `Value Per M$` (right-skewed)
- Correlation matrix with market value
- Average `Overall_Rating` per position (bar chart)

### Task 2 — Preprocessing
- 80/20 train/test split
- Outlier capping via IQR on numerical features (applied only to train set)
- `StandardScaler` for numerical features, `OneHotEncoder` for categoricals via `ColumnTransformer`

### Task 3 — Performance Tier Definition ⭐ *(Mohamed Shaikhoun)*
- Defined 4 tiers based on `Overall_Rating` using domain-aware thresholds:

| Tier | Rating Range |
|------|-------------|
| Low | < 65 |
| Mid | 65 – 74 |
| High | 75 – 84 |
| Elite | ≥ 85 |

- Visualised threshold placement on the rating histogram
- Identified dataset imbalance: majority of players fall in `Low`/`Mid`; `Elite` is rare

### Task 4 — Regression Models
- **Baseline Linear Regression** — MAE, RMSE, R² on train/test
- **Polynomial Regression** — degrees 1–4; degree 3 gave best test generalisation
- **Ridge Regression** — log-spaced alpha sweep
- **Lasso Regression** — automatic feature elimination; identified redundant polynomial features

### Task 5 — Model 2: Logistic Regression
- Multiclass classifier for `Performance_Tier` (`Overall_Rating` excluded to prevent leakage)
- Regularisation sweep over `C` values; L1 vs. L2 comparison
- Confusion matrix and per-class metrics reported

### Task 6 — Model 3: Naïve Bayes Classification ⭐ *(Mohamed Shaikhoun)*
- Trained and compared **three Naïve Bayes variants**:
  - `GaussianNB` — suited to continuous numerical features
  - `BernoulliNB` — suited to binary/one-hot features
  - `ComplementNB` — best choice for the imbalanced dataset; corrects for class-frequency bias
- **Scaling sensitivity test:** confirmed that `StandardScaler` does not affect `GaussianNB` accuracy (linear transforms preserve the shape of each feature's Gaussian, so mean/variance calculations cancel out)
- Justified `ComplementNB` as the most appropriate variant for this dataset

### Task 7 — Cross-Validation
- **Part A:** 5-Fold CV on best regression pipeline (Polynomial deg-3 + Ridge α=0.001) using a `Pipeline` to prevent leakage
- **Part B:** Stratified 5-Fold CV comparing Logistic Regression vs. Complement NB — Logistic Regression wins on both mean accuracy (0.66 vs. 0.62) and stability

### Task 8 — Model Comparison & Regularisation Analysis
- Logistic Regression (C=10, L2) selected as best classifier from Assignment 2
- Analysis of why classification outperforms regression in this setting (broad tiers vs. exact dollar prediction)

---

## 📋 Assignment 3 — Advanced Models & Final System

### Task 1 — Unified Preprocessing Pipeline
- Shared `ColumnTransformer` used consistently across all models to prevent inconsistencies

### Tasks 2–3 — KNN (Classification & Regression)
- Baseline KNN (k=5) + `GridSearchCV` tuning over `n_neighbors`, `weights`, and `metric`
- Sensitivity curve: lower k → overfitting; optimal k found via CV

### Task 4 — SVM / SVR
- `SVC` and `SVR` with `rbf` and `linear` kernels
- Regularisation sweep over `C`; `rbf` consistently outperforms `linear`, confirming non-linearity in the data

### Task 5 — Random Forest
- `RandomForestClassifier` and `RandomForestRegressor` with `GridSearchCV`
- Tuned `n_estimators`, `max_depth`, `min_samples_split`

### Task 6 — XGBoost
- `XGBClassifier` and `XGBRegressor` with `GridSearchCV`
- Tuned `n_estimators`, `learning_rate`, `max_depth`

### Error Diagnosis ⭐ *(Mohamed Shaikhoun)*

Bias/variance learning curves generated for all four model families:

| Model | Task | Diagnosis |
|-------|------|-----------|
| Random Forest Classifier | Performance Tier | Mild **overfitting** — train ~83.5%, CV ~80.5%; visible gap |
| XGBoost Regressor | Market Value | Severe **overfitting** — train RMSE ~1.0, CV RMSE ~2.3 |
| KNN | Both | **High variance** at low k; resolved by increasing k |
| SVM / SVR | Both | **Underfitting** at low C; `rbf` kernel + higher C achieves good fit |

### Model 3 — Final Unified Scouting System 

Built the end-to-end inference pipeline combining all tuned models:

**Committee Models (Ensembles):**
- `VotingRegressor` — combines tuned SVR + Random Forest + XGBoost
- `SoftVotingClassifier` — combines tuned KNN + probability-SVM + Random Forest + XGBoost

**`FinalScoutingSystem` class:**
- Single `.predict()` interface accepting a raw player profile
- Returns predicted market value, performance tier label, confidence score, and per-class probabilities

**Stability Proof:**
- 5-Fold CV on both final ensemble models
- CV mean and std reported to confirm generalisation beyond a single split

**Assignment 2 → 3 Comparison:**
- Final ensemble surpasses Assignment 2 linear/probabilistic baselines on both R² and F1-weighted
- Improvement attributed to: non-linear models, systematic GridSearchCV, model diversity, and cross-validated evaluation

---

## 🏗️ Tech Stack

| Library | Role |
|---|---|
| `pandas`, `numpy` | Data loading and manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing, models, CV, metrics |
| `xgboost` | Gradient boosted trees |

---

## 🚀 How to Run

1. Place `Fifa.csv` in the same directory as the notebook (or upload it in Colab).
2. Open `Assigment_3_ML__F.ipynb` and run all cells top to bottom.
3. `results.json` will be exported automatically at the end with best hyperparameters and CV scores.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook Assigment_3_ML__F.ipynb
```

---

## 📊 Key Results Summary

| System | Task | Metric | Score |
|--------|------|--------|-------|
| Assignment 2 Best (Logistic Regression) | Tier Classification | CV Accuracy | ~0.66 |
| Assignment 3 Final Ensemble | Tier Classification | F1 Weighted | reported in notebook |
| Assignment 3 Final Ensemble | Market Value | R² | reported in notebook |

---

*⭐ = Mohamed's contributions*
