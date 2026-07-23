# Customer Churn Prediction

Portfolio project — memprediksi pelanggan telco yang berpotensi berhenti berlangganan (churn), lengkap dengan interpretasi model dan rekomendasi bisnis.

## 🎯 Tujuan Project

Menjawab pertanyaan bisnis: *pelanggan seperti apa yang paling berisiko churn, dan tindakan retensi apa yang paling relevan untuk masing-masing segmen?*

## 🧰 Tech Stack

- **Python**: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap
- **Visualisasi**: matplotlib, seaborn
- **Dashboard**: Streamlit

## 📁 Struktur Folder

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

## 📊 Dataset

Project ini didesain untuk **Telco Customer Churn** dari Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Cara pakai data asli:**
1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` dari Kaggle
2. Taruh di folder `data/`
3. Di notebook, ubah baris `DATA_PATH` ke file tersebut

> Repo ini sudah menyertakan `data/sample_telco_churn.csv` — dataset sintetis dengan skema kolom identik, dibuat khusus supaya notebook & dashboard bisa langsung dijalankan tanpa perlu download apa pun terlebih dahulu.

## 🔬 Alur Analisis (lihat notebook)

1. **EDA** — distribusi churn, pola berdasarkan tenure, contract, internet service, dll.
2. **Data Cleaning & Feature Engineering** — handling missing value, tenure grouping, encoding.
3. **Handling Imbalanced Data** — SMOTE pada data training.
4. **Modeling** — Logistic Regression, Random Forest, XGBoost.
5. **Evaluasi** — ROC-AUC, Precision-Recall Curve, Confusion Matrix.
6. **Interpretasi Model** — SHAP summary plot untuk memahami fitur paling berpengaruh.
7. **Business Insight** — analisis segmen risiko + rekomendasi retensi konkret.

## 🚀 Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan notebook
jupyter notebook notebooks/01_churn_prediction.ipynb

# 3. Jalankan dashboard
streamlit run app/dashboard.py
```

## 📌 Catatan Penting Sebelum Dipakai Melamar Kerja

Notebook ini sudah **tervalidasi jalan tanpa error** (27 cell, 0 error) menggunakan data sintetis. Sebelum dipakai sebagai portfolio final:

1. **Ganti dataset** ke data asli Kaggle, jalankan ulang seluruh notebook.
2. **Update bagian insight & rekomendasi bisnis** di bagian akhir notebook — angka di versi ini adalah template berdasarkan pola umum, bukan hasil final dari data asli.
3. Tambahkan interpretasi personal di tiap bagian (kenapa milih algoritma X, kenapa metric Y lebih relevan) — ini yang biasanya ditanyakan saat wawancara.
4. Screenshot dashboard atau deploy ke Streamlit Community Cloud (gratis) supaya recruiter bisa akses langsung via link.

## 👤 Author

Dibuat sebagai bagian dari portfolio data science.
