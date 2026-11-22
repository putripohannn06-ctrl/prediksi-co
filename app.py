import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Kadar CO", page_icon="🌫️", layout="centered")

RATA_RATA_CO = 2.15  # rata-rata CO(GT) pada dataset (mg/m³)
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


@st.cache_resource
def muat_model():
    model = joblib.load("model_co.joblib")
    kolom = joblib.load("kolom_fitur.joblib")
    return model, kolom


model, kolom = muat_model()

st.title("🌫️ Prediksi Kadar CO")
st.caption("UAS Big Data dan Machine Learning – Kelompok 8 | Dataset Air Quality UCI")

tab_prediksi, tab_model, tab_tentang = st.tabs(["Prediksi", "Hasil Model", "Tentang Proyek"])

# ---------------------------------------------------------------- TAB 1
with tab_prediksi:
    st.write("Atur bacaan sensor, cuaca, dan waktu, lalu klik **Prediksi** untuk memperkirakan kadar CO(GT) dalam mg/m³.")

    with st.expander("Bacaan sensor", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            s1 = st.slider("PT08.S1(CO)", 600.0, 2100.0, 1075.0)
            s2 = st.slider("PT08.S2(NMHC)", 350.0, 2250.0, 919.0)
            s3 = st.slider("PT08.S3(NOx)", 300.0, 2700.0, 795.0)
            s4 = st.slider("PT08.S4(NO2)", 500.0, 2800.0, 1446.0)
            s5 = st.slider("PT08.S5(O3)", 200.0, 2550.0, 990.0)
        with c2:
            c6h6 = st.slider("C6H6(GT) – benzena", 0.0, 65.0, 8.5)
            nox = st.slider("NOx(GT)", 0.0, 1500.0, 188.0)
            no2 = st.slider("NO2(GT)", 0.0, 350.0, 110.0)

    with st.expander("Cuaca", expanded=False):
        c3, c4, c5 = st.columns(3)
        with c3:
            suhu = st.slider("Suhu T (°C)", -5.0, 45.0, 17.0)
        with c4:
            rh = st.slider("Kelembapan RH (%)", 5.0, 90.0, 49.0)
        with c5:
            ah = st.slider("Kelembapan absolut AH", 0.0, 2.5, 1.0)

    with st.expander("Waktu", expanded=False):
        c6, c7, c8 = st.columns(3)
        with c6:
            jam = st.slider("Jam", 0, 23, 12)
        with c7:
            bulan = st.slider("Bulan", 1, 12, 6)
        with c8:
            hari = st.selectbox("Hari", HARI)

    if st.button("Prediksi", type="primary", use_container_width=True):
        data = {
            "PT08.S1(CO)": s1, "C6H6(GT)": c6h6, "PT08.S2(NMHC)": s2,
            "NOx(GT)": nox, "PT08.S3(NOx)": s3, "NO2(GT)": no2,
            "PT08.S4(NO2)": s4, "PT08.S5(O3)": s5, "T": suhu, "RH": rh, "AH": ah,
            "Hour": jam, "Month": bulan,
            "DayOfWeek": HARI.index(hari),
        }
        x = pd.DataFrame([data])[kolom]  # urutan kolom disamakan dengan saat pelatihan
        hasil = max(float(model.predict(x)[0]), 0.0)

        st.divider()
        st.metric(
            "Perkiraan kadar CO(GT)",
            f"{hasil:.2f} mg/m³",
            delta=f"{hasil - RATA_RATA_CO:+.2f} dari rata-rata dataset",
            delta_color="inverse",  # lebih tinggi = lebih buruk
        )

        if hasil < RATA_RATA_CO * 0.8:
            st.success("Di bawah rata-rata dataset.")
        elif hasil <= RATA_RATA_CO * 1.2:
            st.info("Mirip dengan rata-rata dataset.")
        else:
            st.warning("Di atas rata-rata dataset.")

        st.caption(f"Rata-rata CO di dataset sekitar {RATA_RATA_CO:.2f} mg/m³. "
                   "Model: Linear Regression (R² 0,871 pada data uji).")

# ---------------------------------------------------------------- TAB 2
with tab_model:
    st.subheader("Model yang dipakai")
    m1, m2, m3 = st.columns(3)
    m1.metric("Algoritma", "Linear Regression")
    m2.metric("R² data uji", "0,871")
    m3.metric("MAE", "≈ 0,32 mg/m³")
    st.caption("Evaluasi memakai 20% data terakhir berdasarkan waktu sebagai data uji.")

    st.subheader("Perbandingan lima model (R² data uji)")
    perbandingan = pd.Series({
        "Linear Regression": 0.871,
        "Ridge": 0.871,
        "Gradient Boosting": 0.804,
        "Random Forest": 0.798,
        "Decision Tree": 0.580,
    }).sort_values()
    st.bar_chart(perbandingan)
    st.write("Hubungan antara sensor dan kadar CO ternyata cukup linear, sehingga model sederhana "
             "sudah cukup baik dan model yang lebih kompleks tidak otomatis lebih akurat.")
    st.caption("Random Forest setelah tuning: MAE 0,376, RMSE 0,601, R² 0,800.")

    st.subheader("Pengaruh tiap fitur (koefisien)")
    try:
        # Jika model berupa Pipeline, ambil langkah terakhir
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        koef = pd.Series(est.coef_, index=list(kolom)).sort_values()
        st.bar_chart(koef)
        st.caption("Koefisien dihitung setelah fitur distandarkan (StandardScaler), sehingga besarnya "
                   "bisa dibandingkan antar fitur. Karena banyak fitur saling berkorelasi, "
                   "tanda dan besarnya tetap perlu dibaca hati-hati.")
    except Exception:
        st.info("Koefisien tidak dapat ditampilkan untuk jenis model ini.")

    with st.expander("Fitur yang dipakai model"):
        st.write(", ".join(map(str, kolom)))

# ---------------------------------------------------------------- TAB 3
with tab_tentang:
    st.subheader("Tujuan")
    st.write("Memprediksi kadar karbon monoksida (CO(GT), mg/m³) dari bacaan sensor gas, "
             "data cuaca, dan informasi waktu menggunakan regresi.")

    st.subheader("Dataset")
    st.write("Air Quality dari UCI Machine Learning Repository: 9.357 pengukuran per jam "
             "(Maret 2004 – April 2005), berisi bacaan sensor gas dan pengukuran referensi (berakhiran GT).")

    st.subheader("Pengolahan data")
    st.markdown(
        "- Nilai -200 adalah kode data hilang, diganti menjadi kosong (NaN).\n"
        "- Kolom NMHC(GT) dibuang karena sekitar 90% kosong.\n"
        "- Baris tanpa nilai CO(GT) dibuang, sehingga tersisa 7.674 baris.\n"
        "- Data kosong pada fitur diisi dengan interpolasi.\n"
        "- Dari tanggal dan jam dibuat fitur Hour, Month, dan DayOfWeek."
    )

    st.subheader("Temuan utama")
    st.markdown(
        "- CO paling berkorelasi dengan C6H6(GT), PT08.S2(NMHC), dan PT08.S1(CO); "
        "suhu dan kelembapan hampir tidak berkorelasi.\n"
        "- Pola harian CO punya dua puncak, pagi dan sore, sejalan dengan jam sibuk lalu lintas.\n"
        "- Linear Regression dan Ridge menjadi model terbaik (R² 0,871)."
    )

    st.subheader("Keterbatasan dan saran")
    st.markdown(
        "- Banyak data hilang, data hanya sekitar satu tahun dari satu lokasi, "
        "dan model kurang akurat pada lonjakan CO yang tinggi.\n"
        "- Model memakai juga fitur dari alat referensi (berakhiran GT), sehingga belum murni "
        "hanya memakai sensor murah dan cuaca.\n"
        "- Saran: tambah fitur CO jam sebelumnya, coba model lain seperti XGBoost, "
        "dan uji model tanpa fitur berakhiran (GT)."
    )

    st.subheader("Anggota Kelompok 8")
    st.markdown(
        "- Suci Ramadhani (2417020071)\n"
        "- Nadia Putri Pohan (2417020084)\n"
        "- Fauzan Azim (2417020108)"
    )

    st.caption("UAS Big Data dan Machine Learning – Kelompok 8")
