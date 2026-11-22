import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Kadar CO", page_icon="🌫️")


@st.cache_resource
def muat_model():
    model = joblib.load("model_co.joblib")
    kolom = joblib.load("kolom_fitur.joblib")
    return model, kolom


model, kolom = muat_model()

st.title("Prediksi Kadar CO")
st.caption("UAS Big Data dan Machine Learning – Kelompok 8 | Dataset Air Quality UCI")
st.write("Masukkan bacaan sensor dan data cuaca, lalu klik **Prediksi** untuk memperkirakan kadar CO(GT) dalam mg/m³.")

st.subheader("Bacaan sensor")
c1, c2 = st.columns(2)
with c1:
    s1 = st.number_input("PT08.S1(CO)", 600.0, 2100.0, 1075.0)
    s2 = st.number_input("PT08.S2(NMHC)", 350.0, 2250.0, 919.0)
    s3 = st.number_input("PT08.S3(NOx)", 300.0, 2700.0, 795.0)
    s4 = st.number_input("PT08.S4(NO2)", 500.0, 2800.0, 1446.0)
    s5 = st.number_input("PT08.S5(O3)", 200.0, 2550.0, 990.0)
with c2:
    c6h6 = st.number_input("C6H6(GT) – benzena", 0.0, 65.0, 8.5)
    nox = st.number_input("NOx(GT)", 0.0, 1500.0, 188.0)
    no2 = st.number_input("NO2(GT)", 0.0, 350.0, 110.0)

st.subheader("Cuaca")
c3, c4, c5 = st.columns(3)
with c3:
    suhu = st.number_input("Suhu T (°C)", -5.0, 45.0, 17.0)
with c4:
    rh = st.number_input("Kelembapan relatif RH (%)", 5.0, 90.0, 49.0)
with c5:
    ah = st.number_input("Kelembapan absolut AH", 0.0, 2.5, 1.0)

st.subheader("Waktu")
c6, c7, c8 = st.columns(3)
with c6:
    jam = st.slider("Jam", 0, 23, 12)
with c7:
    bulan = st.slider("Bulan", 1, 12, 6)
with c8:
    hari = st.selectbox("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])

if st.button("Prediksi", type="primary"):
    data = {
        "PT08.S1(CO)": s1, "C6H6(GT)": c6h6, "PT08.S2(NMHC)": s2,
        "NOx(GT)": nox, "PT08.S3(NOx)": s3, "NO2(GT)": no2,
        "PT08.S4(NO2)": s4, "PT08.S5(O3)": s5, "T": suhu, "RH": rh, "AH": ah,
        "Hour": jam, "Month": bulan,
        "DayOfWeek": ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"].index(hari),
    }
    x = pd.DataFrame([data])[kolom]  # urutan kolom disamakan dengan saat pelatihan
    hasil = max(float(model.predict(x)[0]), 0.0)
    st.success(f"Perkiraan kadar CO: **{hasil:.2f} mg/m³**")
    st.caption("Sebagai pembanding, rata-rata CO di dataset sekitar 2,15 mg/m³. "
               "Model: Linear Regression (R² 0,871 pada data uji).")

st.divider()
st.caption("Catatan: model ini memakai juga fitur dari alat referensi (berakhiran GT), "
           "sehingga belum murni hanya memakai sensor murah dan cuaca. Ini keterbatasan proyek.")
