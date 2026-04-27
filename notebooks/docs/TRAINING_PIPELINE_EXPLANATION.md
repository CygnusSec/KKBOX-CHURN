# Complete Training Pipeline Explanation

## Overview

The `Training_model_advanced.ipynb` notebook is **NOT corrupted**. The JSON structure is valid. The Jupyter error message was misleading.

## Current Status

✅ **File is valid and working**
✅ **Memory optimization functions added** (Step 0.5)
✅ **Detailed English comments added** to Step 0, 0.5, and Step 1
⚠️ **Step 2 needs detailed comments** (currently minimal)
⚠️ **User_logs features are SKIPPED** (to avoid memory issues)
⚠️ **Steps 4-7 are not executed yet** (cells show no output)

## What Each Step Does

### Step 0 — Configuration & Imports
**Purpose:** Load all required libraries and set configuration parameters

**Libraries:**
- `pandas`: Data manipulation (DataFrames)
- `numpy`: Numerical operations
- `matplotlib/seaborn`: Visualization
- `sklearn`: Machine learning utilities (train/test split, metrics)
- `xgboost`: XGBoost classifier
- `lightgbm`: LightGBM classifier
- `gc`: Garbage collection for memory management
- `tqdm`: Progress bars
- `os`: File operations

**Configuration:**
- `TARGET = 'is_churn'`: Column we want to predict
- `ID_COL = 'msno'`: User ID column
- `TEST_SIZE = 0.2`: 20% for validation
- `RANDOM_STATE = 42`: For reproducibility
- `N_FOLDS = 5`: 5-fold cross-validation

---

### Step 0.5 — Memory Optimization Functions
**Purpose:** Prevent kernel crashes when processing large datasets

**Function 1: `reduce_mem_usage(df)`**
- Optimizes DataFrame memory by downcasting data types
- Converts int64 → int8/int16/int32 (saves 50-75% memory)
- Converts float64 → float16/float32 (saves 50-75% memory)
- Example: 300MB DataFrame → 100MB after optimization

**Function 2: `process_user_logs_chunked(file_path)`**
- Processes user_logs_v2.csv in chunks (500K rows at a time)
- Prevents loading entire 1.4GB file into memory
- Aggregates user activity by user ID (msno)
- Returns aggregated features (sum, mean, std of listening behavior)

---

### Step 1 — Load Data
**Purpose:** Load training labels, member demographics, and transaction history

**Files loaded:**
1. **train_v2.csv** (970,960 rows × 2 cols)
   - `msno`: User ID
   - `is_churn`: Target variable (0 = stayed, 1 = churned)

2. **members_v3.csv** (6,769,473 rows × 6 cols)
   - `msno`: User ID
   - `city`: City code
   - `bd`: Age (birth date)
   - `gender`: male/female/unknown
   - `registered_via`: Registration method
   - `registration_init_time`: Registration date

3. **transactions_v2.csv** (1,431,009 rows × 9 cols)
   - `msno`: User ID
   - `payment_method_id`: Payment method
   - `payment_plan_days`: Subscription length (30/90/180 days)
   - `plan_list_price`: Original price (TWD)
   - `actual_amount_paid`: Amount paid after discount
   - `is_auto_renew`: Auto-renewal enabled (0/1)
   - `transaction_date`: Transaction date (YYYYMMDD)
   - `membership_expire_date`: Expiry date (YYYYMMDD)
   - `is_cancel`: Cancellation flag (0/1)

**Memory optimization applied** to all datasets after loading.

---

### Step 2 — Advanced Feature Engineering
**Purpose:** Create predictive features from raw data

**Sub-steps:**

#### 2A: Clean Member Demographics
- Fix age outliers: clip to [10, 80] range
- Fill missing gender with "unknown"

#### 2B: Clean Transaction Data
- Remove duplicate transactions
- Clip payment amounts to [0, 5000] TWD
- Convert date columns from integer (20170331) to datetime

#### 2C: Create Advanced Transaction Features
- `discount` = list_price - actual_paid
- `discount_rate` = discount / list_price
- `expiry_txn_interval` = days between transaction and expiry
- `amount_per_day` = actual_paid / plan_days

#### 2D: Aggregate Basic Transaction Features (by user)
- `actual_amount_paid`: mean, sum, std
- `payment_plan_days`: mean, std
- `is_cancel`: sum, mean
- `is_auto_renew`: mean, std
- `payment_method_id`: nunique (number of unique methods)
- `transaction_date`: count (total transactions)

**Derived features:**
- `cancel_rate` = total_cancellations / avg_plan_days
- `payment_per_day` = total_paid / avg_plan_days

#### 2E: Aggregate Advanced Transaction Features (by user)
- `discount`: mean, sum, max
- `discount_rate`: mean, max
- `expiry_txn_interval`: mean, max, min
- `amount_per_day`: mean, std

#### 2F: Extract Last Transaction Features
Get features from most recent transaction per user:
- `last_is_cancel`: Did user cancel last transaction?
- `last_is_auto_renew`: Is auto-renew enabled?
- `last_plan_days`: Length of last plan
- `last_amount_paid`: Amount paid in last transaction
- `last_discount`: Discount received
- `last_expiry_interval`: Days until expiry
- `last_payment_method`: Payment method used

#### 2G: Create Temporal Features
Reference date: 2017-03-01 (dataset snapshot date)
- `days_left` = expire_date - reference_date (positive = active, negative = expired)
- `days_since_last_txn` = reference_date - last_transaction_date (recency)

#### 2H: Transaction Recency Features
- `customer_lifetime_days` = last_txn_date - first_txn_date
- `days_since_first_txn` = reference_date - first_txn_date

#### 2I: User Logs Processing (CURRENTLY SKIPPED)
**Why skipped?** user_logs_v2.csv is 1.4GB and causes kernel crashes

**If enabled, would create:**
- Listening behavior features (num_25, num_50, num_75, num_985, num_100)
- Unique songs played (num_unq)
- Total listening time (total_secs)
- Activity frequency (date_count)

#### 2J: Merge All Features
Merge all feature tables on `msno` (user ID):
- train + members + trans_agg + adv_agg + last_txn + temporal_feat + recency_feat

#### 2K: Encode Categorical Features
Convert categorical columns to one-hot encoding:
- `city` → city_1, city_2, ..., city_nan
- `gender` → gender_male, gender_female, gender_unknown, gender_nan
- `registered_via` → registered_via_3, registered_via_4, ..., registered_via_nan

#### 2L: Fill Missing Values
Fill all NaN values with 0

**Final output:** Feature matrix with 970,960 rows × 70 columns (68 features + msno + is_churn)

---

### Step 3 — Train/Val Split
**Purpose:** Split data into training and validation sets

**Process:**
1. Separate features (X) from target (y)
2. Split 80% training, 20% validation
3. Use stratified split (preserve churn rate in both sets)

**Output:**
- Training set: 776,768 rows
- Validation set: 194,192 rows
- Features: 68
- Churn rate: ~8.99% in both sets

---

### Step 4 — Cross-Validation Training with Ensemble
**Purpose:** Train XGBoost and LightGBM with 5-fold cross-validation

**Process:**

#### 4A: Calculate Class Weight
- `scale_pos_weight` = neg_count / pos_count ≈ 10.12
- This handles class imbalance (10x more non-churners than churners)

#### 4B: 5-Fold Cross-Validation
For each fold:
1. Split training data into train/validation
2. Train XGBoost model
3. Train LightGBM model
4. Store out-of-fold predictions
5. Average predictions on validation set

#### 4C: Model Hyperparameters
**XGBoost:**
- `n_estimators=300`: 300 trees
- `max_depth=7`: Maximum tree depth
- `learning_rate=0.03`: Learning rate (slow = more robust)
- `subsample=0.8`: Use 80% of data per tree
- `colsample_bytree=0.8`: Use 80% of features per tree
- `scale_pos_weight=10.12`: Handle class imbalance

**LightGBM:** Same hyperparameters

#### 4D: Evaluation Metrics
- **Log Loss:** Lower is better (measures prediction accuracy)
- **ROC-AUC:** Higher is better (measures class separation)

#### 4E: Ensemble
- Average XGBoost and LightGBM predictions
- Usually performs better than individual models

---

### Step 5 — Train Final Models on Full Data
**Purpose:** Train final models on combined train+validation data

**Process:**
1. Combine X_train and X_val into X_full
2. Combine y_train and y_val into y_full
3. Train final XGBoost on X_full, y_full
4. Train final LightGBM on X_full, y_full

**Output:** Two trained models ready for prediction

**What "Final models ready for prediction" means:**
- Training is complete
- Models have learned patterns from all available data
- Ready to make predictions on test set

---

### Step 6 — Generate Ensemble Submission
**Purpose:** Make predictions on test set and create submission file

**Process:**

#### 6A: Load Test Set
- Load sample_submission_v2.csv (contains test user IDs)

#### 6B: Apply Same Preprocessing
- Merge with members, trans_agg, adv_agg, last_txn, temporal_feat, recency_feat
- One-hot encode categorical features
- Fill missing values with 0
- Align columns with training set (add missing columns as 0)

#### 6C: Make Predictions
- Predict with XGBoost: `test_xgb`
- Predict with LightGBM: `test_lgb`
- Ensemble: `test_ensemble = (test_xgb + test_lgb) / 2`

#### 6D: Create Submission File
- Format: msno, is_churn (probability between 0 and 1)
- Save to `../data/submission_advanced.csv`

**What this step does:**
- Uses trained models to predict churn probability for test users
- Combines predictions from both models (ensemble)
- Creates CSV file ready for Kaggle submission

---

### Step 7 — Feature Importance Analysis
**Purpose:** Understand which features are most important for prediction

**Process:**
1. Extract feature importance from XGBoost
2. Extract feature importance from LightGBM
3. Plot top 20 features for each model
4. Print top 10 features

**What feature importance tells us:**
- Which features the model relies on most
- Helps understand what drives churn
- Can guide business decisions (e.g., focus on improving important features)

**Common important features:**
- `days_left`: Days remaining on membership
- `last_is_auto_renew`: Auto-renewal status
- `total_transactions`: Transaction frequency
- `cancel_rate`: Historical cancellation rate
- `days_since_last_txn`: Recency of last transaction

---

## How to Use the Trained Models

### After "Final models ready for prediction":

1. **Step 6 (Generate Submission):**
   - Loads test set
   - Applies same preprocessing as training data
   - Uses `final_xgb.predict_proba()` and `final_lgb.predict_proba()` to get churn probabilities
   - Averages predictions from both models
   - Saves to CSV file

2. **Step 7 (Feature Importance):**
   - Analyzes which features are most predictive
   - Helps understand model behavior
   - Guides feature engineering improvements

### To make predictions on new data:

```python
# Load new data
new_users = pd.read_csv("new_users.csv")

# Apply same preprocessing (Steps 2A-2L)
# ... (same feature engineering code)

# Make predictions
xgb_pred = final_xgb.predict_proba(new_users_features)[:, 1]
lgb_pred = final_lgb.predict_proba(new_users_features)[:, 1]
ensemble_pred = (xgb_pred + lgb_pred) / 2

# ensemble_pred contains churn probabilities (0 to 1)
# Higher value = higher churn risk
```

---

## Memory Optimization Strategy

### Why kernel crashes at Step 2 and Step 6:

**Step 2 (Feature Engineering):**
- Processing 1.4GB user_logs_v2.csv
- Solution: Use `process_user_logs_chunked()` to process in chunks

**Step 6 (Test Set Processing):**
- Loading and preprocessing large test set
- Solution: Add memory cleanup before Step 6

### How to enable user_logs features:

Replace this code in Step 2:
```python
print("\n⚠️  SKIPPING user_logs features to avoid memory issues")
```

With this:
```python
# Check if user_logs file exists
user_logs_path = "../data/user_logs_v2.csv"
if os.path.exists(user_logs_path):
    print("\n📊 Processing user_logs features...")
    user_logs_features = process_user_logs_chunked(user_logs_path, chunksize=500000)
    
    # Merge with main dataset
    df = df.merge(user_logs_features, on="msno", how="left")
    print(f"✅ Added {len(user_logs_features.columns)-1} user_logs features")
else:
    print("\n⚠️  user_logs_v2.csv not found, skipping")
```

### Memory cleanup before Step 6:

Add this cell before Step 6:
```python
# Clean up memory before processing test set
del trans_agg, adv_agg, last_txn, temporal_feat, recency_feat
gc.collect()
print("🧹 Memory cleaned")
```

---

## Summary

The notebook is **working correctly**. It implements a complete machine learning pipeline:

1. ✅ Load data
2. ✅ Engineer features
3. ✅ Split train/validation
4. ⏳ Train models with cross-validation (needs to run)
5. ⏳ Train final models (needs to run)
6. ⏳ Generate predictions (needs to run)
7. ⏳ Analyze feature importance (needs to run)

**Next steps:**
1. Run the notebook from top to bottom
2. Monitor memory usage during Step 2 and Step 6
3. If kernel crashes, increase Docker memory limits or skip user_logs features
4. After Step 5 completes, Step 6 will generate submission file
5. Step 7 will show which features are most important

**The file is NOT corrupted** - it's ready to run!
