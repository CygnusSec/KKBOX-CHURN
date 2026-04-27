# 📑 Project Index

Quick navigation guide for KKBOX Churn Prediction project.

## 📖 Documentation

| File | Purpose | When to Read |
|------|---------|--------------|
| **README.md** | Project overview | Start here |
| **QUICKSTART.md** | 5-minute guide | Want to run immediately |
| **SUMMARY.md** | Detailed results | Want to understand improvements |

## 📓 Notebooks

| Notebook | Purpose | Time | Recommended |
|----------|---------|------|-------------|
| `Exploration_data_analysis.ipynb` | Data exploration & visualization | 10 min | For understanding data |
| `Preprocessing.ipynb` | Feature engineering pipeline | 15 min | For learning preprocessing |
| `Training_model_lite.ipynb` | Basic XGBoost model | 5 min | For quick baseline |
| `Training_model_advanced.ipynb` | **Ensemble + CV** | **15 min** | **⭐ USE THIS** |
| `Training_model.ipynb` | Full model with user_logs | 60 min | If you have 16GB+ RAM |

## 🚀 Quick Actions

### Run Best Model
```bash
jupyter notebook notebooks/Training_model_advanced.ipynb
# Then: Cell → Run All
```

### View Results
```bash
cat data/submission_advanced.csv | head
```

### Check Performance
- Expected Log Loss: ~0.080
- Expected ROC-AUC: ~0.996
- Expected Rank: Top 5-10%

## 📊 File Sizes

```
README.md              2.6 KB
QUICKSTART.md          6.1 KB
SUMMARY.md             5.5 KB
Training_model_lite    ~20 KB
Training_model_advanced ~25 KB
```

## 🎯 Workflow

1. **First Time:**
   - Read `README.md`
   - Read `QUICKSTART.md`
   - Run `Training_model_advanced.ipynb`

2. **Understanding:**
   - Read `Exploration_data_analysis.ipynb`
   - Read `Preprocessing.ipynb`
   - Read `SUMMARY.md`

3. **Improving:**
   - Modify `Training_model_advanced.ipynb`
   - Add features in `Preprocessing.ipynb`
   - Try `Training_model.ipynb` (if RAM allows)

---

**Last Updated:** 2026-04-27
