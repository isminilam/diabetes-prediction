import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt

# Load model dan preprocessing
model = joblib.load("model_diabetes.joblib")
scaler = joblib.load("scaler.joblib")
le_gender = joblib.load("le_gender.joblib")
le_smoking = joblib.load("le_smoking.joblib")

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Diabetes", page_icon="🩺", layout="centered")

# Judul dan deskripsi
st.title("Dashboard Diabetes")


# Tabs
tab1, tab2 = st.tabs(["Prediksi Diabetes", "Distribusi Jumlah Penderita Diabetes"])

# ========================
# TAB 1: PREDIKSI DIABETES
# ========================
with tab1:
    st.subheader("🩺 Prediksi Risiko Diabetes")
    st.write("Masukkan informasi berikut untuk memprediksi risiko diabetes.")
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
            age = st.number_input("Usia", min_value=0, max_value=120, value=25)
            hypertension_label = st.selectbox("Riwayat Hipertensi", ["No", "Yes"])
            hypertension = 1 if hypertension_label == "Yes" else 0
            heart_disease_label = st.selectbox("Riwayat Penyakit Jantung", ["No", "Yes"])
            heart_disease = 1 if heart_disease_label == "Yes" else 0

        with col2:
            smoking_status = st.selectbox("Riwayat Merokok", ["No Info", "current", "ever", "former", "never", "not current"])
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.0)
            hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
            glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

        submitted = st.form_submit_button("🔍 Prediksi")

        if submitted:
            try:
                # Encoding
                gender_enc = le_gender.transform([gender])[0]
                smoke_enc = le_smoking.transform([smoking_status])[0]

                # Data numerik untuk diskalakan
                numeric_features = np.array([[age, bmi, hba1c, glucose]])
                numeric_scaled = scaler.transform(numeric_features)

                # Gabungkan semua fitur
                final_features = np.array([[gender_enc, hypertension, heart_disease, smoke_enc]])
                input_combined = np.concatenate([final_features, numeric_scaled], axis=1)

                # Prediksi
                prediction = model.predict(input_combined)[0]

                # Hasil
                if prediction == 1:
                    st.error("⚠️ Hasil: **Positif Diabetes**")
                else:
                    st.success("✅ Hasil: **Negatif Diabetes**")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

    st.markdown("""
    **📌 Keterangan Riwayat Merokok:**
    - `No Info` : Tidak ada informasi tentang kebiasaan merokok.
    - `current` : Saat ini aktif merokok.
    - `ever` : Pernah merokok setidaknya sekali dalam hidup.
    - `former` : Pernah merokok secara rutin, tetapi telah berhenti.
    - `never` : Tidak pernah merokok sama sekali.
    - `not current` : Pernah merokok, tetapi saat ini tidak merokok. Ini bisa termasuk mantan perokok ringan atau yang merokok tidak secara teratur.
    """)

# ======================================
# TAB 2: VISUALISASI DATA JUMLAH KASUS
# ======================================
with tab2:
    st.subheader("📊 Jumlah Penderita Diabetes di Jawa Barat")

    try:
        df = pd.read_csv("cleaned_data_penderita_dm.csv")
        df.columns = [col.strip() for col in df.columns]  # Hilangkan spasi
        df["tahun"] = df["tahun"].astype(str)
        df["jumlah_penderita_dm"] = pd.to_numeric(df["jumlah_penderita_dm"], errors="coerce")

        st.dataframe(df, use_container_width=True)

# Visualisasi tambahan: Bar chart per tahun
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("tahun", title="Tahun", sort=None),
            y=alt.Y("jumlah_penderita_dm", title="Jumlah Penderita"),
            tooltip=["tahun", "jumlah_penderita_dm"],
            color=alt.value("#4C78A8")
        ).properties(
            title="Distribusi Jumlah Penderita Diabetes per Tahun (Bar Chart)",
            width=700,
            height=400
        ).configure_title(fontSize=18, anchor='start')

        st.altair_chart(bar_chart, use_container_width=True)


    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
