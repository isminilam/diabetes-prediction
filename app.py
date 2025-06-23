import streamlit as st
import numpy as np
import pandas as pd
import joblib

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
tab1, tab2 = st.tabs(["Prediksi Diabetes", "Kenali Diabetes"])

# ========================
# TAB 1: PREDIKSI DIABETES
# ========================
with tab1:
    st.subheader("🩺 Prediksi Risiko Diabetes Tipe 2")
    st.write("Masukkan informasi berikut untuk memprediksi risiko diabetes.")

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Jenis Kelamin", ["", "Male", "Female"])
            age = st.number_input("Usia", min_value=1, max_value=120)
            hypertension_label = st.selectbox("Riwayat Hipertensi", ["", "No", "Yes"])
            heart_disease_label = st.selectbox("Riwayat Penyakit Jantung", ["", "No", "Yes"])

        with col2:
            smoking_status = st.selectbox("Riwayat Merokok", ["", "No Info", "current", "ever", "former", "never", "not current"])
            bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=80.0)
            hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=50.0)
            glucose_input = st.text_input("Blood Glucose Level (mg/dL)", "")

        submitted = st.form_submit_button("🔍 Prediksi")

        if submitted:
            # Validasi input kosong
            if (
                gender == "" or
                hypertension_label == "" or
                heart_disease_label == "" or
                smoking_status == "" or
                glucose_input.strip() == "" or
                bmi == 0.0 or
                hba1c == 0.0
            ):
                st.warning("⚠️ Harap lengkapi semua isian sebelum melakukan prediksi.")
            else:
                try:
                    glucose = float(glucose_input)

                    # Encoding
                    gender_enc = le_gender.transform([gender])[0]
                    smoke_enc = le_smoking.transform([smoking_status])[0]
                    hypertension = 1 if hypertension_label == "Yes" else 0
                    heart_disease = 1 if heart_disease_label == "Yes" else 0

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

                except ValueError:
                    st.error("❌ Blood Glucose harus berupa angka.")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat memproses prediksi: {e}")

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
# TAB 2: KENALI DIABETES
# ======================================
with tab2:
    st.header("🔍 Kenali Diabetes")

    # Topik 1: Apa Itu Diabetes
    with st.expander("📌 Apa Itu Diabetes?"):
        st.markdown("""
        Diabetes adalah kondisi kronis yang terjadi ketika tubuh tidak dapat memproduksi cukup insulin atau tidak dapat menggunakan insulin secara efektif, sehingga menyebabkan kadar glukosa (gula) dalam darah meningkat (hiperglikemia).
        """)

    # Topik 2: Dua Tipe Diabetes
    with st.expander("🧬 Dua Tipe Diabetes"):
        st.markdown("""
        **Diabetes Tipe 1** merupakan jenis utama diabetes yang dapat dialami pada segala usia, tetapi sering terjadi pada anak-anak dan orang dewasa. Penderita diabetes tipe 1 memerlukan insulin seumur hidup untuk bertahan hidup karena tubuh mereka tidak dapat memproduksi insulin.

        **Diabetes Tipe 2** mencakup sebagian besar kasus diabetes di seluruh dunia, yaitu lebih dari **90%**. Diabetes tipe 2 dapat dicegah atau ditunda kemunculannya melalui perubahan gaya hidup sehat. Bahkan, dapat membaik atau dikendalikan sepenuhnya, jika ditangani sejak awal.
        """)


    # Topik 3: Gejala Umum Diabetes
    with st.expander("⚠️ Gejala Umum Diabetes"):
        st.markdown("""
        Beberapa gejala umum yang sering dialami penderita diabetes antara lain:

        - Sering buang air kecil
        - Rasa haus yang berlebihan
        - Berat badan turun drastis
        - Luka sulit sembuh
        - Mudah lelah
        - Penglihatan kabur
        """)
    
    # Topik 4: Statistik Penderita Diabetes
    with st.expander("📊 Statistik Penderita Diabetes"):
        st.markdown("""
        #### 🌍 **Di Dunia**
        - Menurut *International Diabetes Federation (IDF) Atlas*, pada tahun 2024 diperkirakan terdapat sekitar **588,7 juta orang dewasa** (usia 20–79 tahun) di seluruh dunia yang hidup dengan diabetes.  
        - Tanpa upaya pencegahan yang serius, jumlah ini diperkirakan akan meningkat menjadi **852,5 juta jiwa pada tahun 2050**.

        #### 🇮🇩 **Di Indonesia**
        - Masih berdasarkan data IDF, Indonesia menempati **peringkat ke-5** negara dengan jumlah penderita diabetes terbanyak di dunia pada tahun 2024.  
        - Tercatat ada sekitar **20,4 juta orang dewasa** (usia 20–79 tahun) yang mengidap diabetes, dan angka ini diprediksi meningkat menjadi **28,6 juta jiwa pada tahun 2050**.  
        - Lebih memprihatinkan lagi, sekitar **73,2% kasus belum terdiagnosis**, sehingga banyak penderita yang tidak mendapatkan penanganan medis secara tepat dan dini.
        """)

    # Topik 5: Source
    with st.expander("🔗 Sumber"):
        st.markdown("""
        - [WHO – Diabetes Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/diabetes)  
        - [IDF Diabetes Atlas 2024](https://diabetesatlas.org/)  
        """)

