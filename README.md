# Customer Churn Prediction

Portfolio project — memprediksi pelanggan telco yang berpotensi berhenti berlangganan (churn), lengkap dengan interpretasi model dan rekomendasi bisnis.

## 🔗 Live Demo
Dashboard interaktif: https://customer-churn-prediction-chx2rwx5d4m8avjnhaxnmr.streamlit.app

##  Tujuan Project

Menjawab pertanyaan bisnis: *pelanggan seperti apa yang paling berisiko churn, dan tindakan retensi apa yang paling relevan untuk masing-masing segmen?*

##  Tech Stack

- **Python**: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap
- **Visualisasi**: matplotlib, seaborn
- **Dashboard**: Streamlit

##  Struktur Folder

```
customer-churn-portfolio/
├── data/
│   └── sample_telco_churn.csv      # data sintetis untuk demo (ganti dengan data Kaggle asli)
├── notebooks/
│   └── 01_churn_prediction.ipynb   # analisis end-to-end
├── app/
│   └── dashboard.py                # dashboard interaktif Streamlit
├── requirements.txt
└── README.md
```

##  Dataset

Project ini didesain untuk **Telco Customer Churn** dari Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Cara pakai data asli:**
1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` dari Kaggle
2. Taruh di folder `data/`
3. Di notebook, ubah baris `DATA_PATH` ke file tersebut

> Repo ini sudah menyertakan `data/sample_telco_churn.csv` — dataset sintetis dengan skema kolom identik, dibuat khusus supaya notebook & dashboard bisa langsung dijalankan tanpa perlu download apa pun terlebih dahulu.

##  Alur Analisis (lihat notebook)

1. **EDA** — distribusi churn, pola berdasarkan tenure, contract, internet service, dll.
2. **Data Cleaning & Feature Engineering** — handling missing value, tenure grouping, encoding.
3. **Handling Imbalanced Data** — SMOTE pada data training.
4. **Modeling** — Logistic Regression, Random Forest, XGBoost.
5. **Evaluasi** — ROC-AUC, Precision-Recall Curve, Confusion Matrix.
6. **Interpretasi Model** — SHAP summary plot untuk memahami fitur paling berpengaruh.
7. **Business Insight** — analisis segmen risiko + rekomendasi retensi konkret.

##  Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan notebook
jupyter notebook notebooks/01_churn_prediction.ipynb

# 3. Jalankan dashboard
streamlit run app/dashboard.py
```

