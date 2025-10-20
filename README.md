

## ‎(Datasets)‎
اختر **Dat
- **Pima Indians Diabetes** — (Binary) — Target ≥ 75%  
  Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## ‎(start)‎

```bash
# 1) إنشاء بيئة
python -m venv .venv
source .venv/bin/activate  # 
pip install -r requirements.txt

## ‎هيكل المشروع (Required Structure)‎
```
fcnn-classification-project/
  README.md
  requirements.txt
  .gitignore
  data/
    raw/
    processed/ (train.csv, val.csv, test.csv, scaler.pkl)
  notebooks/
    01_EDA_and_preprocessing.ipynb
    02_model_training.ipynb
    03_hyperparameter_tuning.ipynb
  models/
    model_1.h5  model_2.h5  model_3.h5  best_model.h5  training_history.json
  backend/
    main.py  models.py  preprocessing.py  requirements.txt  Dockerfile  README.md
  frontend/
    index.html  style.css  script.js
  assets/
  visualizations/
    eda_plots/  training_curves/  confusion_matrices/  comparison_plots/
  docs/
    model_comparison.md  hyperparameter_analysis.md  
