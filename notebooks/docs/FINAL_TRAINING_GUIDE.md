# Final Complete Training Model - KKBOX Churn Prediction

## Overview
This guide combines **memory optimization** + **complete preprocessing** + **advanced feature engineering** + **ensemble training** into one comprehensive workflow.

## 📋 Complete Workflow

### Phase 1: Memory Optimization & Data Loading
### Phase 2: Data Cleaning & Preprocessing  
### Phase 3: Advanced Feature Engineering (with user_logs)
### Phase 4: Model Training (XGBoost + LightGBM Ensemble)
### Phase 5: Cross-Validation & Evaluation
### Phase 6: Submission Generation

---

## 🚀 Quick Start

### Option A: Use the Optimized Notebook
Open `notebooks/Training_model_advanced.ipynb` - it already includes:
- ✅ Memory-efficient data loading
- ✅ Complete preprocessing pipeline
- ✅ 68 engineered features
- ✅ XGBoost + LightGBM ensemble
- ✅ 5-fold cross-validation
- ✅ Automatic submission generation

**Current limitation**: Skips user_logs to avoid memory issues

### Option B: Add user_logs Features (Recommended)
Follow these steps to include user behavior features:

#### Step 1: Process user_logs with Memory Optimization
```bash
# Open Jupyter
# Run: notebooks/00_Memory_Optimization_Helper.ipynb
```

This will create: `data/user_logs_features_aggregated.csv` (~200MB instead of 28GB)

#### Step 2: Modify Training_model_advanced.ipynb

Add this cell after Step 2 (after loading data):

```python
# Load aggregated user_logs features
print("Loading user_logs features...")
user_logs_features = pd.read_csv("../data/user_logs_features_aggregated.csv")
user_logs_features = reduce_mem_usage(user_logs_features)
print(f"User logs features: {user_logs_features.shape}")
```

Then in Step 2 (Feature Engineering), add user_logs merge:

```python
# After other merges, add:
df = df.merge(user_logs_features, on="msno", how="left")
```

---

## 📊 Complete Feature Set

### Current Features (68 total):

#### 1. **Member Demographics** (5 features)
- `bd` - Age (clipped 10-80)
- `city_*` - One-hot encoded cities
- `gender_*` - One-hot encoded (male/female/unknown)
- `registered_via_*` - Registration method
- `registration_init_time` - Registration date

#### 2. **Transaction Aggregations** (7 features)
- `actual_amount_paid_mean` - Average payment
- `actual_amount_paid_sum` - Total spent
- `actual_amount_paid_std` - Payment variability
- `payment_plan_days_mean` - Average plan length
- `payment_plan_days_std` - Plan variability
- `is_cancel_sum` - Total cancellations
- `is_auto_renew_mean` - Auto-renew rate

#### 3. **Advanced Transaction Features** (11 features)
- `discount_mean/sum/max` - Discount patterns
- `discount_rate_mean/max` - Discount percentage
- `expiry_txn_interval_mean/max/min` - Time between transaction and expiry
- `amount_per_day_mean/std` - Daily spending rate
- `payment_method_id_nunique` - Number of different payment methods

#### 4. **Derived Features** (2 features)
- `cancel_rate` - Cancellations per day
- `payment_per_day` - Average daily spend

#### 5. **Last Transaction Features** (7 features)
- `last_is_cancel` - Did user cancel last time?
- `last_is_auto_renew` - Auto-renew on last transaction?
- `last_plan_days` - Last plan duration
- `last_amount_paid` - Last payment amount
- `last_discount` - Last discount received
- `last_expiry_interval` - Days until last expiry
- `last_payment_method` - Last payment method used

#### 6. **Temporal Features** (2 features)
- `days_left` - Days remaining on membership
- `days_since_last_txn` - Recency of last transaction

#### 7. **Recency Features** (2 features)
- `customer_lifetime_days` - How long user has been customer
- `days_since_first_txn` - Days since first transaction

#### 8. **User Logs Features** (32 features - when added)
Per user aggregations:
- `num_25_sum/mean/std` - Songs <25% played
- `num_50_sum/mean/std` - Songs 25-50% played
- `num_75_sum/mean/std` - Songs 50-75% played
- `num_985_sum/mean/std` - Songs 75-98.5% played
- `num_100_sum/mean/std` - Songs >98.5% played
- `num_unq_sum/mean/std` - Unique songs
- `total_secs_sum/mean/std` - Total listening time
- `date_count/min/max` - Activity days and date range

**Total with user_logs: 100 features**

---

## 🎯 Model Performance

### Current Results (without user_logs):
```
XGBoost  CV Log Loss: 0.0754
LightGBM CV Log Loss: 0.0772
ENSEMBLE Log Loss:    0.0767
ROC-AUC:              0.9961
```

### Expected with user_logs:
```
Estimated improvement: 5-10%
Expected Log Loss: ~0.068-0.072
Expected ROC-AUC: ~0.997+
```

---

## 💾 Memory Management Strategy

### Problem:
- `user_logs.csv`: 28GB (too large)
- `user_logs_v2.csv`: 1.3GB (still large)
- Available RAM: 12GB (Docker limit)

### Solution:
1. **Chunked Reading**: Process in 500K row chunks
2. **Immediate Aggregation**: Aggregate each chunk by user
3. **Memory Optimization**: Reduce data types (40-60% savings)
4. **Result**: 28GB → 200MB (99.3% reduction)

### Implementation:
```python
def reduce_mem_usage(df):
    """Optimize data types to reduce memory"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                # ... etc
    return df
```

---

## 🔧 Hyperparameters

### XGBoost:
```python
XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10.12,  # Handle class imbalance
    eval_metric='logloss',
    random_state=42
)
```

### LightGBM:
```python
LGBMClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10.12,
    random_state=42
)
```

### Ensemble:
```python
# Simple average of predictions
ensemble_pred = (xgb_pred + lgb_pred) / 2
```

---

## 📈 Training Strategy

### 1. **Stratified K-Fold Cross-Validation** (5 folds)
- Maintains class distribution in each fold
- Reduces overfitting
- More reliable performance estimates

### 2. **Out-of-Fold Predictions**
- Each fold predicts on its validation set
- Combines to form full training set predictions
- Used for model selection

### 3. **Final Model Training**
- Train on full dataset (train + validation)
- Use for test set predictions
- Maximizes use of available data

---

## 📁 Output Files

### Generated Files:
1. `data/user_logs_features_aggregated.csv` - Processed user logs
2. `data/submission_advanced.csv` - Final predictions
3. Feature importance plots
4. Performance metrics

---

## 🎓 Best Practices Applied

### Data Quality:
- ✅ Outlier clipping (age, payment amounts)
- ✅ Missing value handling (gender → 'unknown')
- ✅ Duplicate removal
- ✅ Date parsing and validation

### Feature Engineering:
- ✅ Aggregations (mean, sum, std, count)
- ✅ Ratios and derived features
- ✅ Temporal features (recency, lifetime)
- ✅ Last transaction features
- ✅ Behavioral patterns (user_logs)

### Model Training:
- ✅ Class imbalance handling (scale_pos_weight)
- ✅ Cross-validation for robustness
- ✅ Ensemble for better predictions
- ✅ Feature importance analysis

### Memory Optimization:
- ✅ Chunked reading for large files
- ✅ Data type optimization
- ✅ Garbage collection
- ✅ Efficient aggregation

---

## 🚦 Step-by-Step Execution

### Complete Workflow:

```bash
# 1. Start Docker with optimized memory
docker-compose down
docker-compose up -d

# 2. Open Jupyter
open http://localhost:8888

# 3. Process user_logs (one-time, ~10-30 minutes)
# Run: notebooks/00_Memory_Optimization_Helper.ipynb
# Output: data/user_logs_features_aggregated.csv

# 4. Train final model
# Run: notebooks/Training_model_advanced.ipynb
# (Modify to include user_logs features as shown above)
# Output: data/submission_advanced.csv

# 5. Submit to Kaggle
# Upload: data/submission_advanced.csv
```

---

## 📊 Expected Timeline

| Task | Time | Memory Peak |
|------|------|-------------|
| Process user_logs | 10-30 min | ~8GB |
| Load & preprocess data | 2-5 min | ~4GB |
| Feature engineering | 1-2 min | ~5GB |
| Train XGBoost (5-fold) | 5-10 min | ~6GB |
| Train LightGBM (5-fold) | 5-10 min | ~6GB |
| Generate submission | 1-2 min | ~4GB |
| **Total** | **25-60 min** | **~8GB peak** |

---

## 🎯 Performance Targets

### Minimum (without user_logs):
- Log Loss: < 0.080
- ROC-AUC: > 0.995

### Target (with user_logs):
- Log Loss: < 0.070
- ROC-AUC: > 0.997

### Stretch (with hyperparameter tuning):
- Log Loss: < 0.065
- ROC-AUC: > 0.998

---

## 🔍 Troubleshooting

### Issue: Kernel crashes during training
**Solution**: 
- Restart Docker: `docker-compose restart`
- Reduce chunk size in user_logs processing
- Use `user_logs_v2.csv` instead of `user_logs.csv`

### Issue: Out of memory error
**Solution**:
- Run `gc.collect()` between steps
- Use `reduce_mem_usage()` on all dataframes
- Increase Docker memory limit in `docker-compose.yml`

### Issue: Low model performance
**Solution**:
- Ensure user_logs features are included
- Check for data leakage
- Verify feature engineering logic
- Try hyperparameter tuning

---

## 📚 Files Reference

### Notebooks:
- `00_Memory_Optimization_Helper.ipynb` - Process large files
- `Training_model_advanced.ipynb` - Main training pipeline
- `Preprocessing.ipynb` - Data cleaning reference
- `Exploration_data_analysis.ipynb` - EDA reference

### Data Files:
- `train_v2.csv` - Training labels
- `members_v3.csv` - User demographics
- `transactions_v2.csv` - Transaction history
- `user_logs_v2.csv` - User behavior logs
- `sample_submission_v2.csv` - Submission template

### Configuration:
- `docker-compose.yml` - Docker memory settings
- `requirements.txt` - Python dependencies

---

## ✅ Checklist

Before training:
- [ ] Docker running with 12GB memory
- [ ] All CSV files in `data/` folder
- [ ] `user_logs_features_aggregated.csv` created
- [ ] Jupyter accessible at localhost:8888

During training:
- [ ] No memory warnings
- [ ] Cross-validation scores reasonable
- [ ] Feature importance makes sense
- [ ] Submission file generated

After training:
- [ ] Submission file has correct format
- [ ] Predicted churn rate ~0.08-0.10
- [ ] No missing values in submission
- [ ] File size ~50MB

---

**Status**: ✅ Complete and ready to use
**Last Updated**: April 27, 2026
**Version**: 1.0 - Final Complete Training Model
