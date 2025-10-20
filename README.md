
# ‎مشروع FCNN للتصنيف (End‑to‑End)‎

> ‎هذا المستودع يحقق متطلبات "FCNN Technical Project Exam" بنسخة جاهزة للتشغيل محليًا والنشر.‎  
> ‎المصطلحات والتعابير التقنية بالـ English تُترك كما هي؛ الشرح والتنظيم بالعربية (RTL).‎

## ‎البيانات (Datasets)‎
اختر **Dataset** واحد (روابط موثوقة):

- **Heart Disease (Cleveland)** — (Binary) — Target ≥ 80%  
  Kaggle: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci  
  UCI: https://archive.ics.uci.edu/dataset/45/heart+disease

- **Telco Customer Churn** — (Binary) — Target ≥ 78%  
  Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- **Pima Indians Diabetes** — (Binary) — Target ≥ 75%  
  Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

- **Bank Marketing** — (Binary) — Target ≥ 88%  
  UCI: https://archive.ics.uci.edu/dataset/222/bank+marketing

- **Wine Quality** — (Multi‑class 3–8) — Target ≥ 65%  
  UCI: https://archive.ics.uci.edu/dataset/186/wine+quality

> ‎بعد تحميل الملف ضع الـ CSV داخل `data/raw/` ثم نفّذ خطوات الـ EDA/Preprocessing.‎

## ‎تشغيل سريع (Quickstart)‎

```bash
# 1) إنشاء بيئة
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) تثبيت الاعتمادات
pip install -r requirements.txt

# 3) EDA + Preprocessing
python data_preprocessing.py --dataset telco --input data/raw/telco.csv

# 4) تدريب النماذج (Keras FCNN)
python train_fcnn.py

# 5) اختيار أفضل نموذج وإنشاء ملفات: models/best_model.h5 و data/processed/scaler.pkl

# 6) تشغيل الواجهة الخلفية FastAPI
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 7) فتح الواجهة الأمامية
# افتح frontend/index.html في المتصفح (أو قدمها عبر خادم بسيط)
```

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
    model_comparison.md  hyperparameter_analysis.md  api_documentation.md
```

## ‎نصائح الدرجات (Grading Hints)‎
- ‎أكمل 3 تصميمات FCNN مختلفة + Regularization (Dropout/L1/L2) + BatchNorm.‎
- ‎جرّب Optimizers (Adam/SGD/RMSProp) و Schedulers.‎
- ‎قدّم Confusion Matrices + Classification Reports.‎
- ‎بناء FastAPI مع `/health`, `/predict`, `/predict_batch` + CORS + Swagger.‎
- ‎واجهة Frontend متجاوبة + CSV Download للنتائج.‎
- ‎نشر عبر Docker محليًا أو Cloud (Railway/Render + Vercel/Netlify).‎
