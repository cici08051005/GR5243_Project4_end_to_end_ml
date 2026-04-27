# Pipeline Enhancement -- Changelog

## Summary

Enhanced all 3 notebooks: fixed data quality issues in cleaning, added missing value
documentation, engineered better features, switched to KNN imputation, prevented data
leakage, and added SMOTENC for class imbalance handling. Every column drop is justified
with a chart or table in the notebook.

---

## Problems Fixed

### 1. Removed 6 Constant Columns (100% Missing in Raw Data)

The following columns were previously selected and silently imputed, but they are
**100% missing** in the raw data. After imputation they become constant-valued
features with zero variance and zero predictive power:

| Column | Imputed Value | Unique Values After |
|--------|--------------|---------------------|
| `host_response_rate` | 0.0 | 1 |
| `host_acceptance_rate` | 0.0 | 1 |
| `host_response_time` | "Unknown" | 1 |
| `host_since` | NaN | 0 |
| `host_total_listings_count` | 0.0 | 1 |
| `instant_bookable` | 0.0 | 1 |

**Justification**: Missing value summary table and bar chart in notebook Section 5-6.

### 2. Replaced `bathrooms` with Parsed `bathrooms_text`

| Column | Missing Rate | Info |
|--------|-------------|------|
| `bathrooms` (old) | 40.2% | Numeric only |
| `bathrooms_text` (source) | 0.3% | "1 bath", "1 shared bath", etc. |

Parsed into two new features:
- `bathrooms_count` (float): number of bathrooms
- `bathroom_is_shared` (0/1): whether bathroom is shared

**Justification**: Side-by-side comparison table in notebook Section 8.

### 3. Added Outlier Capping

| Column | Old Max | 99th Percentile | Capped At |
|--------|---------|-----------------|-----------|
| `minimum_nights` | 1,124 | 90 | 90 |
| `maximum_nights` | 2,147,483,648 | 1,125 | 1,125 |

**Justification**: Before/after box plots in notebook Section 12.

### 4. KNN Imputation Replaces Median/Mode

Switched from global median/mode imputation to **KNN imputation (k=5)** using
`sklearn.impute.KNNImputer`. KNN leverages inter-feature correlations (e.g.,
bedrooms/beds/accommodates) to produce more realistic fill values.

| Column Group | Old Method | New Method |
|-------------|-----------|-----------|
| Numeric (bedrooms, beds, etc.) | Global median | KNN (k=5) |
| Binary (host_is_superhost, etc.) | Mode | KNN (k=5), rounded to 0/1 |
| Review scores | Global median | KNN (k=5) |
| reviews_per_month | Fill 0 | Fill 0 (domain logic, unchanged) |
| Categoricals | "Unknown" | "Unknown" (unchanged) |

---

## New Features Added

### 4. Amenity Features (from `amenities`, 0% missing)

- `amenity_count`: total number of amenities listed
- `has_wifi`: binary flag
- `has_ac`: binary flag (air conditioning)
- `has_kitchen`: binary flag
- `has_washer`: binary flag

### 5. Text Length Features

- `name_length`: character count of listing title (0% missing)
- `description_length`: character count of listing description (4.1% missing, filled with 0)

### 6. Date Features Excluded (Data Leakage Prevention)

- `first_review` and `last_review` are **intentionally excluded** from the pipeline
- Features derived from these dates (e.g., `listing_age_days`, `days_since_last_review`) are post-outcome proxies for booking activity
- `days_since_last_review` had 0.51 correlation with target and dominated model importance (35% in RF) — clear data leakage

### 7. Trust Signal

- `host_has_profile_pic`: binary (0/1), 1.4% missing in raw data

---

## Data Quality Documentation Added

- **Section 5**: Missing value summary table + bar chart for all 85 raw columns
- **Section 6**: Justification table for dropping 6 constant columns
- **Section 15**: Before vs after data quality comparison table

---

## Output Changes

| Metric | Before | After |
|--------|--------|-------|
| Rows | 36,445 | 36,445 |
| Columns | 54 | ~57 |
| Missing values | 0 | 0 |
| Constant columns | 5 | 0 |

File: `data/processed/nyc_airbnb_cleaned.csv` (overwritten)

---

## Downstream Impact

### Notebook 02 (`02_eda_unsupervised.ipynb`)
- `cluster_features` list updated: removed dead columns and leaky date features, added new features
- `corr_cols` list updated: same removals, added `name_length`, `description_length`
- PCA and KMeans re-run with new feature set

### Notebook 03 (`03_feature_engineering_modeling.ipynb`)
- `identifier_cols` updated: removed `host_since` (no longer in cleaned data)
- Categorical/numeric feature detection is automatic via `select_dtypes`
- **Models replaced**: LR and HGB → SVM (RBF) and PyTorch MLP; RF kept
- **SMOTENC** for RF and MLP; `class_weight="balanced"` for SVM (O(n²) constraint)
- **PyTorch MLP** with BatchNorm, Dropout, early stopping, training curve visualization
- **SHAP analysis** added: TreeExplainer on RF, summary + bar plots
- Uses `imblearn.pipeline.Pipeline` so SMOTENC is applied only to training folds

---

## Modeling Changes (Notebook 03)

### 8. SMOTENC for Class Imbalance (Pre-computed)

The target is imbalanced (72.2% low vs 27.8% high occupancy). Added **SMOTENC**
(Synthetic Minority Over-sampling for Nominal and Continuous) to generate synthetic
minority samples in training data.

- **Pre-computed once** on raw training data before preprocessing — all 3 models
  receive the same balanced, preprocessed dataset (no redundant recomputation)
- Training set: 29,156 → 42,124 samples (21,062 per class)
- Preprocessing (StandardScaler + OneHotEncoder) fitted on SMOTENC'd data, applied to both train and test
- 339 features after one-hot encoding
- `k_neighbors=5` for synthetic sample generation

### 9. Data Leakage Prevention

Removed post-outcome proxy variables from model inputs:

| Variable Group | Reason |
|---------------|--------|
| `availability_30/60/90/365/eoy` | Remaining availability decreases with bookings |
| `number_of_reviews*`, `reviews_per_month` | Direct measures of booking activity |
| `days_since_last_review`, `listing_age_days` | Temporal proxies for booking recency |
| `review_scores_*_missing` (7 cols) | 100% of high-occupancy listings have reviews |

### 10. Model Replacement: Three Learning Paradigms

Replaced Logistic Regression and Hist Gradient Boosting with SVM and PyTorch MLP:

| Model | Old | New | Paradigm |
|-------|-----|-----|----------|
| Model 1 | Logistic Regression | **Random Forest** (tuned) | Bagging ensemble |
| Model 2 | Hist Gradient Boosting | **SVM (RBF kernel)** (tuned) | Kernel-based margin classifier |
| Model 3 | — | **PyTorch MLP** | Deep neural network |

**Random Forest details**: Hyperparameter tuning via `RandomizedSearchCV` (30 iterations,
3-fold stratified CV). Best params: n_estimators=300, max_depth=None, min_samples_leaf=3,
min_samples_split=10, max_features=0.3. CV AUC: 0.965.

**SVM details**: Tuned via `RandomizedSearchCV` (20 iterations) on 10K stratified subsample
(O(n²) complexity makes full-data tuning impractical). Best params: C=5.0, gamma=scale.
Retrained on full 42K SMOTENC'd data. Uses `class_weight="balanced"` and `decision_function`
for ROC-AUC scoring.

**PyTorch MLP details**:
- Architecture: Input(339) → 256 → 128 → 64 → 1 with BatchNorm, ReLU, Dropout
- Training: Adam optimizer (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau scheduler
- Early stopping (patience=15) on validation loss (15% holdout from SMOTENC'd training data)
- 129,153 trainable parameters, converged at epoch 46

### 11. SHAP Values Analysis

Added SHAP (SHapley Additive exPlanations) interpretability analysis:
- `TreeExplainer` on Random Forest (1000-sample subset for efficiency)
- Beeswarm summary plot: shows direction and magnitude of each feature's impact
- Bar plot: mean absolute SHAP value per feature
- New figures: `16_shap_summary_plot.png`, `17_shap_bar_plot.png`

### 12. Hyperparameter Tuning

| Model | Method | Search Space | Iterations | CV Folds |
|-------|--------|-------------|-----------|----------|
| Random Forest | `RandomizedSearchCV` | n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features | 30 | 3 |
| SVM (RBF) | `RandomizedSearchCV` on 10K subsample | C, gamma | 20 | 3 |
| PyTorch MLP | Manual (early stopping + LR scheduling) | — | — | — |

### Final Model Performance

| Model | AUC | F1 | Accuracy | Precision | Recall |
|-------|-----|-----|----------|-----------|--------|
| Random Forest (tuned) | 0.931 | 0.756 | 0.867 | 0.770 | 0.742 |
| PyTorch MLP | 0.903 | 0.728 | 0.846 | 0.714 | 0.742 |
| SVM (RBF, tuned) | 0.899 | 0.716 | 0.837 | 0.694 | 0.740 |

Best model by AUC: **Random Forest** (0.931, CV AUC 0.965 ± 0.001)
