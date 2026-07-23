"""
Dashboard Streamlit — Customer Churn Prediction
Jalankan dengan: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

RANDOM_STATE = 42


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


@st.cache_resource
def train_model(df):
    df_model = df.drop(columns=["customerID"]).copy()
    df_model["Churn"] = df_model["Churn"].map({"Yes": 1, "No": 0})
    cat_cols = df_model.select_dtypes(include="object").columns.tolist()
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    X = df_model.drop(columns=["Churn"])
    y = df_model["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    model.fit(X_train_res, y_train_res)

    return model, X.columns.tolist(), X_test, y_test


st.title("Customer Churn Prediction Dashboard")
st.caption("Rumah Digicraft — Data Science Portfolio Project")

data_path = st.sidebar.text_input(
    "Path dataset (CSV)", value="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

try:
    df = load_data(data_path)
except FileNotFoundError:
    st.error(f"File tidak ditemukan: {data_path}. Sesuaikan path di sidebar.")
    st.stop()

model, feature_cols, X_test, y_test = train_model(df)

tab1, tab2, tab3 = st.tabs(["Overview", "Segment Analysis", "Prediksi Individual"])

with tab1:
    col1, col2, col3 = st.columns(3)
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    col1.metric("Total Pelanggan", f"{len(df):,}")
    col2.metric("Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Rata-rata Tenure", f"{df['tenure'].mean():.1f} bulan")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        df["Churn"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
        ax.set_title("Distribusi Churn")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots()
        churn_by_contract = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
        churn_by_contract.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#C44E52")
        ax.set_title("Churn Rate (%) berdasarkan Contract")
        st.pyplot(fig)

with tab2:
    st.subheader("Churn Rate berdasarkan Kombinasi Segmen")
    seg_col1 = st.selectbox("Segmen 1", ["Contract", "InternetService", "PaymentMethod"], index=0)
    seg_col2 = st.selectbox("Segmen 2", ["InternetService", "Contract", "TechSupport"], index=0)

    seg_df = (
        df.groupby([seg_col1, seg_col2])["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="ChurnRate(%)")
        .sort_values("ChurnRate(%)", ascending=False)
    )
    st.dataframe(seg_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = seg_df.pivot(index=seg_col1, columns=seg_col2, values="ChurnRate(%)")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Simulasi Prediksi Churn untuk 1 Pelanggan")
    st.caption("Isi profil pelanggan untuk melihat probabilitas churn.")

    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Tenure (bulan)", 0, 72, 12)
    monthly_charges = c2.slider("Monthly Charges", 18.0, 120.0, 70.0)
    contract = c3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    c4, c5, c6 = c1, c2, c3
    internet = c4.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = c5.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    payment = c6.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

    if st.button("Prediksi Churn"):
        # Bangun 1 baris input yang sesuai kolom hasil one-hot encoding
        input_dict = dict.fromkeys(feature_cols, 0)
        input_dict["tenure"] = tenure
        input_dict["MonthlyCharges"] = monthly_charges
        input_dict["TotalCharges"] = monthly_charges * tenure

        for col, val in [
            (f"Contract_{contract}", 1),
            (f"InternetService_{internet}", 1),
            (f"TechSupport_{tech_support}", 1),
            (f"PaymentMethod_{payment}", 1),
        ]:
            if col in input_dict:
                input_dict[col] = val

        input_df = pd.DataFrame([input_dict])[feature_cols]
        proba = model.predict_proba(input_df)[0][1]

        st.metric("Probabilitas Churn", f"{proba * 100:.1f}%")
        if proba > 0.5:
            st.warning("⚠️ Pelanggan ini berisiko tinggi churn. Pertimbangkan program retensi.")
        else:
            st.success("✅ Pelanggan ini relatif aman dari risiko churn.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Dashboard ini bagian dari portfolio Data Science — "
    "prediksi churn pelanggan menggunakan Random Forest + SMOTE."
)
