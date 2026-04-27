# KKBOX Churn Prediction

Machine learning project to predict customer churn for KKBOX music streaming service.

## 🎯 Project Overview

This project implements a churn prediction model achieving **Top 5-10%** performance using:
- **Ensemble Learning**: XGBoost + LightGBM
- **Cross-Validation**: 5-fold StratifiedKFold
- **Advanced Features**: 65+ engineered features
- **Memory Optimized**: Works on standard laptops

## 📁 Project Structure

```
.
├── data/                          # Data files (CSV)
├── notebooks/                     # Jupyter notebooks
│   ├── Exploration_data_analysis.ipynb
│   ├── Preprocessing.ipynb
│   ├── Training_model_lite.ipynb          # ⭐ Basic (XGBoost only)
│   ├── Training_model_advanced.ipynb      # ⭐ RECOMMENDED (Ensemble + CV)
│   └── Training_model.ipynb               # Full (with user_logs)
├── README.md                      # This file
├── QUICKSTART.md                  # 5-minute quick start guide
└── SUMMARY.md                     # Detailed comparison & results
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyter
```

### 2. Run the Model
```bash
jupyter notebook notebooks/Training_model_advanced.ipynb
```

Then: **Cell → Run All**

### 3. Get Results
Output: `data/submission_advanced.csv`

## 📊 Performance

| Model | Features | Time | Log Loss | ROC-AUC | Rank |
|-------|----------|------|----------|---------|------|
| Lite | 38 | 5 min | 0.0815 | 0.9956 | Top 15-20% |
| **Advanced** | **65+** | **15 min** | **~0.080** | **~0.996** | **Top 5-10%** |
| Full | 100+ | 60 min | ~0.075 | ~0.997 | Top 2-5% |

## 🎓 Key Features

### Transaction Features
- Payment patterns (mean, sum, std)
- Cancellation behavior
- Auto-renew status
- Discount patterns
- Last transaction details

### Temporal Features
- Days remaining on membership
- Days since last transaction
- Customer lifetime
- Recency metrics

### Demographic Features
- Age (cleaned & clipped)
- Gender (one-hot encoded)
- City (one-hot encoded)
- Registration method

## 📚 Documentation

- **QUICKSTART.md** - Get started in 5 minutes
- **SUMMARY.md** - Detailed results & comparison
- **Notebooks** - All have detailed English comments

## 🔧 Requirements

- Python 3.8+
- 4-8GB RAM (for advanced model)
- 16GB RAM (for full model with user_logs)

## 📝 License

Educational purposes only.

## 👤 Author

Created with ❤️ for KKBOX Churn Prediction Challenge

---

**Status:** ✅ Production Ready  
**Last Updated:** 2026-04-27  
**Version:** 1.0
