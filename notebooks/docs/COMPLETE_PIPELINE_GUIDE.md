# Complete KKBOX Churn Prediction Pipeline Guide

**From EDA to Model Training and Deployment**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Training](#5-model-training)
6. [Model Evaluation](#6-model-evaluation)
7. [Making Predictions](#7-making-predictions)
8. [Using the Trained Model](#8-using-the-trained-model)

---

## 1. Project Overview

### 1.1 Business Problem
**Goal:** Predict which KKBOX music streaming users will churn (cancel subscription) in the next month.

**Why it matters:**
- Retaining existing customers is cheaper than acquiring new ones
- Early identification allows targeted retention campaigns
- Understanding churn drivers helps improve service

### 1.2 Dataset Description

**Three main data files:**

1. **train_v2.csv** (970,960 rows × 2 columns)
   - `msno`: User ID (hashed)
   - `is_churn`: Target variable (0 = stayed, 1 = churned)

2. **members_v3.csv** (6,769,473 rows × 6 columns)
   - `msno`: User ID
   - `city`: City code
   - `bd`: Age (birth date)
   - `gender`: male/female/unknown
   - `registered_via`: Registration method (3, 4, 7, 9, 13)
   - `registration_init_time`: Registration timestamp

3. **transactions_v2.csv** (1,431,009 rows × 9 columns)
   - `msno`: User ID
   - `payment_method_id`: Payment method
   - `payment_plan_days`: Subscription length (30/90/180 days)
   - `plan_list_price`: Original price (TWD)
   - `actual_amount_paid`: Amount paid after discount
   - `is_auto_renew`: Auto-renewal enabled (0/1)
   - `transaction_date`: Transaction date (YYYYMMDD)
   - `membership_expire_date`: Expiry date (YYYYMMDD)
   - `is_cancel`: Cancellation flag (0/1)

4. **user_logs_v2.csv** (18,396,362 rows × 9 columns) - Optional
   - User listening behavior (songs played, listening time, etc.)
   - Very large file (1.4GB) - requires memory optimization

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Target Variable Analysis

**Churn Rate:**
```
Churn rate: 8.99%
- Not churned (0): 883,470 users (91.01%)
- Churned (1):      87,490 users (8.99%)
```

**Key insight:** Highly imbalanced dataset - need to handle class imbalance during training.

### 2.2 Data Quality Assessment

**Missing Values:**
- `members.gender`: ~1.5% missing → Fill with "unknown"
- `members.bd`: No missing, but has outliers

**Outliers:**
- `bd` (age): Range from -7168 to 2016 (clearly errors)
  - Solution: Clip to reasonable range [10, 80]
- `actual_amount_paid`: Some negative values
  - Solution: Clip to [0, 5000] TWD

**Duplicates:**
- `transactions`: Contains duplicate rows
  - Solution: Remove with `.drop_duplicates()`

### 2.3 Demographic Features Analysis

**Age Distribution:**
- Most users: 20-40 years old
- Churn rate slightly higher for very young (<20) and older (>50) users

**Gender Distribution:**
- Male: ~55%
- Female: ~43%
- Unknown: ~2%
- Churn rate similar across genders

**City Distribution:**
- Top cities: 1, 4, 5, 13, 22
- Some cities have slightly higher churn rates

**Registration Method:**
- Methods: 3, 4, 7, 9, 13
- Method 7 has highest churn rate (~12%)
- Method 9 has lowest churn rate (~7%)

### 2.4 Transaction Behavior Analysis

**Payment Plan Days:**
- 30 days: Highest churn rate (~15%)
- 90 days: Medium churn rate (~8%)
- 180 days: Lowest churn rate (~5%)
- **Insight:** Longer commitments = lower churn

**Auto-Renewal:**
- Auto-renew enabled: ~3% churn rate
- Auto-renew disabled: ~12% churn rate
- **Insight:** Auto-renewal is strong predictor of retention

**Cancellation History:**
- Users who canceled before: Much higher churn rate
- **Insight:** Past behavior predicts future behavior

**Transaction Frequency:**
- More transactions = lower churn
- Single transaction users: Highest churn risk

**Discount Behavior:**
- Heavy discount users: Slightly higher churn
- **Insight:** Price-sensitive users more likely to churn

### 2.5 Temporal Analysis

**Days Until Expiry:**
- Negative days (expired): Very high churn (~40%)
- 0-30 days: High churn (~15%)
- 30+ days: Low churn (~3%)
- **Insight:** Membership status is critical predictor

**Recency (Days Since Last Transaction):**
- Recent transaction (<30 days): Low churn
- Old transaction (>90 days): High churn
- **Insight:** Engagement recency matters

### 2.6 Key EDA Findings

**Strong Churn Predictors:**
1. Days left on membership (negative = expired)
2. Auto-renewal status
3. Payment plan length
4. Transaction frequency
5. Cancellation history
6. Days since last transaction

**Weak Predictors:**
- Gender
- City (some variation but not strong)
- Age (some variation at extremes)

---

## 3. Data Preprocessing

### 3.1 Clean Member Demographics

```python
# Fix age outliers: clip to reasonable range [10, 80]
members["bd"] = members["bd"].clip(10, 80)

# Fill missing gender with "unknown"
members["gender"] = members["gender"].fillna("unknown")
```

**Why:**
- Ages outside [10, 80] are data errors
- Keeping "unknown" as category preserves information

### 3.2 Clean Transaction Data

```python
# Remove duplicate transactions
transactions = transactions.drop_duplicates()

# Clip payment amounts to valid range [0, 5000]
transactions["actual_amount_paid"] = transactions["actual_amount_paid"].clip(0, 5000)

# Convert date columns from integer to datetime
transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"].astype(str), 
    format="%Y%m%d", 
    errors="coerce"
)
transactions["membership_expire_date"] = pd.to_datetime(
    transactions["membership_expire_date"].astype(str), 
    format="%Y%m%d", 
    errors="coerce"
)
```

**Why:**
- Duplicates inflate aggregated features
- Negative payments are impossible
- Datetime format enables temporal calculations

---

## 4. Feature Engineering

### 4.1 Transaction-Derived Features

**Discount Features:**
```python
# Discount amount: how much money was saved
transactions["discount"] = transactions["plan_list_price"] - transactions["actual_amount_paid"]

# Discount rate: percentage of discount
transactions["discount_rate"] = transactions["discount"] / (transactions["plan_list_price"] + 1)
```

**Temporal Features:**
```python
# Days between transaction and membership expiry
transactions["expiry_txn_interval"] = (
    transactions["membership_expire_date"] - transactions["transaction_date"]
).dt.days

# Daily spending rate
transactions["amount_per_day"] = transactions["actual_amount_paid"] / (transactions["payment_plan_days"] + 1)
```

### 4.2 Aggregated Transaction Features

**Basic Aggregations (by user):**
```python
trans_agg = transactions.groupby("msno").agg({
    "actual_amount_paid": ["mean", "sum", "std"],  # Spending patterns
    "payment_plan_days":  ["mean", "std"],          # Plan length patterns
    "is_cancel":          ["sum", "mean"],          # Cancellation history
    "is_auto_renew":      ["mean", "std"],          # Auto-renewal behavior
    "payment_method_id":  ["nunique"],              # Payment method diversity
    "transaction_date":   ["count"]                 # Transaction frequency
})
```

**Derived Features:**
```python
# Cancellation rate per day
trans_agg["cancel_rate"] = trans_agg["is_cancel_sum"] / (trans_agg["payment_plan_days_mean"] + 1)

# Average daily spending
trans_agg["payment_per_day"] = trans_agg["actual_amount_paid_sum"] / (trans_agg["payment_plan_days_mean"] + 1)
```

### 4.3 Last Transaction Features

```python
# Get features from most recent transaction
last_txn = (
    transactions
    .sort_values("transaction_date")
    .groupby("msno").tail(1)
    [["msno", "is_cancel", "is_auto_renew", "payment_plan_days", 
      "actual_amount_paid", "discount", "expiry_txn_interval"]]
    .rename(columns={
        "is_cancel": "last_is_cancel",
        "is_auto_renew": "last_is_auto_renew",
        "payment_plan_days": "last_plan_days",
        "actual_amount_paid": "last_amount_paid",
        "discount": "last_discount",
        "expiry_txn_interval": "last_expiry_interval"
    })
)
```

**Why:** Most recent behavior best reflects current user state.

### 4.4 Temporal Features

```python
REFERENCE_DATE = pd.to_datetime("2017-03-01")  # Dataset snapshot date

# Days left on membership
last_trans["days_left"] = (last_trans["membership_expire_date"] - REFERENCE_DATE).dt.days

# Days since last transaction (recency)
last_trans["days_since_last_txn"] = (REFERENCE_DATE - last_trans["transaction_date"]).dt.days

# Customer lifetime
txn_dates["customer_lifetime_days"] = (txn_dates["last_txn_date"] - txn_dates["first_txn_date"]).dt.days
```

### 4.5 Categorical Encoding

```python
# One-hot encode categorical features
df = pd.get_dummies(df, columns=["city", "gender", "registered_via"], dummy_na=True)
```

**Result:** Each category becomes a binary column (0 or 1).

Example:
- `gender="male"` → `gender_male=1, gender_female=0, gender_unknown=0`

### 4.6 Handle Missing Values

```python
# Fill all missing values with 0
df = df.fillna(0)
```

**Why:** After merging, some users may not have transactions → NaN values.

### 4.7 Final Feature Set

**Total features: 68-70 (depending on categorical cardinality)**

Categories:
- Demographics: 6 features (age, gender, city, registration)
- Transaction aggregations: 20 features
- Last transaction: 7 features
- Temporal: 4 features
- One-hot encoded: 30-40 features

---

## 5. Model Training

### 5.1 Train/Validation Split

```python
X = df.drop(columns=["is_churn", "msno"])  # Features
y = df["is_churn"]                          # Target

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,        # 80% train, 20% validation
    random_state=42,      # Reproducibility
    stratify=y            # Preserve churn rate in both sets
)
```

**Result:**
- Training: 776,768 rows
- Validation: 194,192 rows
- Churn rate: 8.99% in both sets

### 5.2 Handle Class Imbalance

```python
# Calculate class weight
neg_count = (y_train == 0).sum()  # Non-churners
pos_count = (y_train == 1).sum()  # Churners
scale_pos_weight = neg_count / pos_count  # ≈ 10.12
```

**Why:** With 91% non-churners and 9% churners, model would achieve 91% accuracy by always predicting "no churn". We need to give more weight to the minority class.

### 5.3 Model Selection

**Two models used:**

1. **XGBoost (Extreme Gradient Boosting)**
   - Tree-based ensemble method
   - Builds trees sequentially, each correcting previous errors
   - Excellent for tabular data
   - Handles non-linear relationships well

2. **LightGBM (Light Gradient Boosting Machine)**
   - Similar to XGBoost but faster
   - Uses leaf-wise tree growth (vs level-wise in XGBoost)
   - Better memory efficiency

**Why ensemble?** Combining predictions from both models usually performs better than either alone.

### 5.4 Hyperparameters Explained

```python
XGBClassifier(
    n_estimators=300,      # Number of trees to build
    max_depth=7,           # Maximum tree depth (controls complexity)
    learning_rate=0.03,    # Step size (lower = more robust but slower)
    subsample=0.8,         # Use 80% of data per tree (prevents overfitting)
    colsample_bytree=0.8,  # Use 80% of features per tree
    scale_pos_weight=10.12,# Weight for positive class (handles imbalance)
    eval_metric='logloss', # Metric to optimize
    random_state=42,       # Reproducibility
    n_jobs=-1              # Use all CPU cores
)
```

**Parameter Tuning Guide:**

| Parameter | Lower Value | Higher Value | Typical Range |
|-----------|-------------|--------------|---------------|
| `n_estimators` | Underfitting | Better performance (but slower) | 100-500 |
| `max_depth` | Simpler model | More complex (risk overfitting) | 3-10 |
| `learning_rate` | More robust | Faster convergence | 0.01-0.3 |
| `subsample` | More regularization | Less regularization | 0.5-1.0 |
| `colsample_bytree` | More regularization | Less regularization | 0.5-1.0 |

### 5.5 Cross-Validation Training

**5-Fold Stratified Cross-Validation:**

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]
    X_vl = X_train.iloc[valid_idx]
    y_vl = y_train.iloc[valid_idx]
    
    # Train model
    model.fit(X_tr, y_tr)
    
    # Store predictions
    predictions[valid_idx] = model.predict_proba(X_vl)[:, 1]
```

**Why cross-validation?**
- More reliable performance estimate
- Uses all data for both training and validation
- Reduces overfitting risk

**How it works:**
1. Split training data into 5 folds
2. Train on 4 folds, validate on 1 fold
3. Repeat 5 times (each fold used for validation once)
4. Average results across all folds

### 5.6 Training Process

**Step-by-step:**

1. **Initialize models** with hyperparameters
2. **For each fold:**
   - Split data into train/validation
   - Train XGBoost on train set
   - Predict on validation set
   - Train LightGBM on train set
   - Predict on validation set
   - Store predictions
3. **Calculate metrics** across all folds
4. **Train final models** on full training data
5. **Make predictions** on test set

---

## 6. Model Evaluation

### 6.1 Evaluation Metrics

**1. Log Loss (Logarithmic Loss)**
```
Log Loss = -1/N * Σ [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
```

- **Range:** 0 to ∞ (lower is better)
- **Interpretation:** Measures how well predicted probabilities match actual labels
- **Good value:** < 0.15 for this problem
- **Why use it:** Penalizes confident wrong predictions heavily

**2. ROC-AUC (Area Under ROC Curve)**
```
ROC curve: True Positive Rate vs False Positive Rate at different thresholds
AUC: Area under this curve
```

- **Range:** 0 to 1 (higher is better)
- **Interpretation:** Probability that model ranks random positive higher than random negative
- **Good value:** > 0.85 for this problem
- **Why use it:** Threshold-independent, good for imbalanced data

### 6.2 Typical Results

**Cross-Validation Results:**
```
XGBoost:
  - CV Log Loss: 0.0815
  - Validation ROC-AUC: 0.9956

LightGBM:
  - CV Log Loss: 0.0820
  - Validation ROC-AUC: 0.9954

Ensemble (Average):
  - Log Loss: 0.0810
  - ROC-AUC: 0.9958
```

**Interpretation:**
- Log Loss ~0.08: Excellent calibration
- ROC-AUC ~0.996: Excellent discrimination
- Ensemble performs slightly better than individual models

### 6.3 Feature Importance

**Top 10 Most Important Features:**

1. **days_left** (0.15-0.20 importance)
   - Days remaining on membership
   - Negative = expired (very high churn risk)

2. **last_is_auto_renew** (0.10-0.15)
   - Whether auto-renewal is enabled
   - Strong predictor of retention

3. **total_transactions** (0.08-0.12)
   - Number of transactions
   - More transactions = more engaged

4. **cancel_rate** (0.06-0.10)
   - Historical cancellation rate
   - Past behavior predicts future

5. **days_since_last_txn** (0.05-0.08)
   - Recency of last transaction
   - Recent activity = lower churn

6. **actual_amount_paid_sum** (0.04-0.07)
   - Total amount spent
   - Higher value customers less likely to churn

7. **payment_per_day** (0.03-0.06)
   - Daily spending rate
   - Indicates commitment level

8. **last_plan_days** (0.03-0.05)
   - Length of last subscription
   - Longer plans = lower churn

9. **is_auto_renew_mean** (0.02-0.04)
   - Average auto-renewal rate
   - Consistency matters

10. **payment_plan_days_mean** (0.02-0.04)
    - Average plan length
    - Longer commitments = retention

**Business Insights:**
- Focus retention on users with negative days_left
- Encourage auto-renewal (strong predictor)
- Monitor users with increasing days_since_last_txn
- Reward high-value users (high spending)

---

## 7. Making Predictions

### 7.1 Load Test Set

```python
test = pd.read_csv("../data/sample_submission_v2.csv")
# Contains msno (user IDs) for test users
```

### 7.2 Apply Same Preprocessing

**CRITICAL:** Test set must have exact same features as training set.

```python
# Merge with same feature tables
df_test = test[["msno"]].merge(members, on="msno", how="left")
df_test = df_test.merge(trans_agg, on="msno", how="left")
df_test = df_test.merge(adv_agg, on="msno", how="left")
df_test = df_test.merge(last_txn, on="msno", how="left")
df_test = df_test.merge(temporal_feat, on="msno", how="left")
df_test = df_test.merge(recency_feat, on="msno", how="left")

# One-hot encode
df_test = pd.get_dummies(df_test, columns=["city", "gender", "registered_via"], dummy_na=True)

# Fill missing
df_test = df_test.fillna(0)

# Align columns with training set
for col in train_cols:
    if col not in df_test.columns:
        df_test[col] = 0
X_test = df_test[train_cols]
```

### 7.3 Generate Predictions

```python
# Predict with XGBoost
test_xgb = final_xgb.predict_proba(X_test)[:, 1]

# Predict with LightGBM
test_lgb = final_lgb.predict_proba(X_test)[:, 1]

# Ensemble (simple average)
test_ensemble = (test_xgb + test_lgb) / 2
```

**Output:** Churn probability for each user (0 to 1)

### 7.4 Create Submission File

```python
submission = pd.DataFrame({
    "msno": test["msno"],
    "is_churn": test_ensemble
})

submission.to_csv("submission_advanced.csv", index=False)
```

**Format:**
```
msno,is_churn
user1,0.0234
user2,0.8765
user3,0.1234
...
```

---

## 8. Using the Trained Model

### 8.1 Load Saved Model

```python
import pickle

# Save models
with open('final_xgb.pkl', 'wb') as f:
    pickle.dump(final_xgb, f)
with open('final_lgb.pkl', 'wb') as f:
    pickle.dump(final_lgb, f)

# Load models
with open('final_xgb.pkl', 'rb') as f:
    final_xgb = pickle.load(f)
with open('final_lgb.pkl', 'rb') as f:
    final_lgb = pickle.load(f)
```

### 8.2 Predict for New Users

```python
def predict_churn(user_data):
    """
    Predict churn probability for new users.
    
    Parameters:
    -----------
    user_data : pandas.DataFrame
        Must contain same features as training data
    
    Returns:
    --------
    predictions : numpy.array
        Churn probabilities (0 to 1)
    """
    # Apply same preprocessing
    user_data = preprocess(user_data)  # Your preprocessing function
    
    # Predict with both models
    xgb_pred = final_xgb.predict_proba(user_data)[:, 1]
    lgb_pred = final_lgb.predict_proba(user_data)[:, 1]
    
    # Ensemble
    ensemble_pred = (xgb_pred + lgb_pred) / 2
    
    return ensemble_pred

# Example usage
new_users = pd.read_csv("new_users.csv")
churn_probs = predict_churn(new_users)

# Identify high-risk users
high_risk = new_users[churn_probs > 0.5]
print(f"High-risk users: {len(high_risk)}")
```

### 8.3 Interpret Predictions

**Probability Thresholds:**

| Probability | Risk Level | Action |
|-------------|------------|--------|
| 0.0 - 0.2 | Low | Monitor |
| 0.2 - 0.5 | Medium | Engage with content |
| 0.5 - 0.7 | High | Offer discount/promotion |
| 0.7 - 1.0 | Very High | Personal outreach |

**Example:**
```python
def categorize_risk(prob):
    if prob < 0.2:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    elif prob < 0.7:
        return "High"
    else:
        return "Very High"

new_users["risk_level"] = churn_probs.apply(categorize_risk)
new_users["churn_probability"] = churn_probs
```

### 8.4 Business Actions

**Based on predictions:**

1. **Low Risk (< 0.2):**
   - Continue normal service
   - Monitor for changes

2. **Medium Risk (0.2 - 0.5):**
   - Send engagement emails
   - Recommend personalized playlists
   - Highlight new features

3. **High Risk (0.5 - 0.7):**
   - Offer 20% discount on renewal
   - Provide free trial of premium features
   - Survey for feedback

4. **Very High Risk (> 0.7):**
   - Personal phone call or email
   - Offer 50% discount or free month
   - Priority customer support

### 8.5 Model Monitoring

**Track over time:**

```python
# Monthly churn prediction accuracy
monthly_predictions = []
monthly_actuals = []

for month in months:
    preds = predict_churn(users_in_month)
    actuals = get_actual_churn(users_in_month)
    
    monthly_predictions.append(preds)
    monthly_actuals.append(actuals)
    
    # Calculate metrics
    auc = roc_auc_score(actuals, preds)
    print(f"{month}: AUC = {auc:.4f}")
```

**When to retrain:**
- AUC drops below 0.90
- Log loss increases above 0.15
- Business metrics (retention rate) change significantly
- New features become available

### 8.6 A/B Testing

**Test retention strategies:**

```python
# Identify high-risk users
high_risk_users = users[churn_probs > 0.5]

# Randomly split into control and treatment
control = high_risk_users.sample(frac=0.5, random_state=42)
treatment = high_risk_users.drop(control.index)

# Apply intervention to treatment group
send_discount_offer(treatment)

# Measure results after 30 days
control_churn_rate = control["churned"].mean()
treatment_churn_rate = treatment["churned"].mean()

print(f"Control churn rate: {control_churn_rate:.2%}")
print(f"Treatment churn rate: {treatment_churn_rate:.2%}")
print(f"Reduction: {(control_churn_rate - treatment_churn_rate):.2%}")
```

---

## Summary

### Complete Pipeline

1. **EDA:** Understand data, identify patterns, find strong predictors
2. **Preprocessing:** Clean data, handle outliers and missing values
3. **Feature Engineering:** Create predictive features from raw data
4. **Model Training:** Train XGBoost and LightGBM with cross-validation
5. **Evaluation:** Assess performance with log loss and ROC-AUC
6. **Prediction:** Generate churn probabilities for new users
7. **Action:** Implement retention strategies based on predictions

### Key Takeaways

**Strong Predictors:**
- Membership status (days_left)
- Auto-renewal status
- Transaction frequency
- Cancellation history
- Engagement recency

**Model Performance:**
- Log Loss: ~0.08 (excellent)
- ROC-AUC: ~0.996 (excellent)
- Ensemble outperforms individual models

**Business Impact:**
- Early identification of at-risk users
- Targeted retention campaigns
- Improved customer lifetime value
- Reduced churn rate

### Files Reference

- **EDA:** `notebooks/Exploration_data_analysis.ipynb`
- **Preprocessing:** `notebooks/Preprocessing.ipynb`
- **Training:** `notebooks/Training_model_advanced.ipynb`
- **Memory Optimization:** `notebooks/00_Memory_Optimization_Helper.ipynb`
- **Documentation:** `notebooks/docs/TRAINING_PIPELINE_EXPLANATION.md`

---

**End of Guide**
