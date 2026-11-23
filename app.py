import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Kadar CO", page_icon="🌫️", layout="centered")

HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
RATA_CO = 2.15  # rata-rata CO(GT) pada dataset (mg/m³)

# (label, min, max, step, nama kolom)
SENSOR = [
    ("PT08.S1(CO)", 600.0, 2100.0, 5.0, "PT08.S1(CO)"),
    ("PT08.S2(NMHC)", 350.0, 2250.0, 5.0, "PT08.S2(NMHC)"),
    ("PT08.S3(NOx)", 300.0, 2700.0, 5.0, "PT08.S3(NOx)"),
    ("PT08.S4(NO2)", 500.0, 2800.0, 5.0, "PT08.S4(NO2)"),
    ("PT08.S5(O3)", 200.0, 2550.0, 5.0, "PT08.S5(O3)"),
]
GAS = [
    ("C6H6(GT) – benzena", 0.0, 65.0, 0.1, "C6H6(GT)"),
    ("NOx(GT)", 0.0, 1500.0, 1.0, "NOx(GT)"),
    ("NO2(GT)", 0.0, 350.0, 1.0, "NO2(GT)"),
]
CUACA = [
    ("Suhu T (°C)", -5.0, 45.0, 0.1, "T"),
    ("Kelembapan relatif RH (%)", 5.0, 90.0, 0.5, "RH"),
    ("Kelembapan absolut AH", 0.0, 2.5, 0.01, "AH"),
]

# Skenario contoh: nilai rata-rata/tengah dari data asli pada jam tertentu
PRESET = {
    "Nilai tengah data": {
        "PT08.S1(CO)": 1063.0, "C6H6(GT)": 8.2, "PT08.S2(NMHC)": 909.0, "NOx(GT)": 180.0,
        "PT08.S3(NOx)": 805.0, "NO2(GT)": 109.0, "PT08.S4(NO2)": 1463.0, "PT08.S5(O3)": 963.0,
        "T": 17.8, "RH": 49.5, "AH": 1.0, "Hour": 12, "Month": 6, "Hari": "Rabu"},
    "Jam sibuk sore (±19.00)": {
        "PT08.S1(CO)": 1282.0, "C6H6(GT)": 17.7, "PT08.S2(NMHC)": 1201.0, "NOx(GT)": 364.0,
        "PT08.S3(NOx)": 664.0, "NO2(GT)": 150.0, "PT08.S4(NO2)": 1693.0, "PT08.S5(O3)": 1296.0,
        "T": 19.8, "RH": 44.6, "AH": 1.03, "Hour": 19, "Month": 6, "Hari": "Rabu"},
    "Dini hari sepi (±05.00)": {
        "PT08.S1(CO)": 892.0, "C6H6(GT)": 3.1, "PT08.S2(NMHC)": 641.0, "NOx(GT)": 87.0,
        "PT08.S3(NOx)": 1107.0, "NO2(GT)": 60.0, "PT08.S4(NO2)": 1231.0, "PT08.S5(O3)": 713.0,
        "T": 14.2, "RH": 60.9, "AH": 1.04, "Hour": 5, "Month": 6, "Hari": "Rabu"},
}

st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; max-width: 900px;}
.hero {background: linear-gradient(135deg, #0B0F14 0%, #14532D 100%); color: #fff;
       padding: 28px 30px; border-radius: 18px; margin-bottom: 18px;}
.hero h1 {margin: 0 0 6px 0; font-size: 2rem; color: #fff;}
.hero p {margin: 0; color: #BBF7D0; font-size: 0.95rem;}
.badge {display:inline-block; background:#22C55E; color:#0B0F14; font-weight:700;
        padding: 3px 12px; border-radius: 999px; font-size: 0.78rem; margin-bottom: 10px;}
.card {background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:18px 20px; margin-bottom:12px;}
.result {background:#fff; border:2px solid #22C55E; border-radius:18px; padding:22px; text-align:center;}
.result .big {font-size: 3.2rem; font-weight: 800; color:#14532D; line-height:1.1;}
.result .unit {color:#64748B; font-size:1rem;}
.gauge {position:relative; height:14px; border-radius:999px;
        background: linear-gradient(90deg,#86EFAC 0%,#FDE68A 40%,#FCA5A5 100%); margin: 18px 6px 6px 6px;}
.mark {position:absolute; top:-6px; width:6px; height:26px; background:#0B0F14; border-radius:3px;}
.small {color:#64748B; font-size:0.85rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <span class="badge">DEMO · UAS BIG DATA DAN MACHINE LEARNING</span>
  <h1>Prediksi Kadar CO</h1>
  <p>Memperkirakan CO(GT) dari bacaan sensor dan cuaca · Dataset Air Quality UCI · Kelompok 8</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def muat_model():
    return joblib.load("model_co.joblib"), joblib.load("kolom_fitur.joblib")


try:
    model, kolom = muat_model()
except Exception as e:  # file model tidak ditemukan / versi library berbeda
    st.error(f"Model gagal dimuat: {e}")
    st.stop()


def terapkan_preset():
    p = PRESET[st.session_state["skenario"]]
    for k, v in p.items():
        st.session_state["in_" + k] = v


if "in_Hour" not in st.session_state:
    p = PRESET["Nilai tengah data"]
    for k, v in p.items():
        st.session_state["in_" + k] = v

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Hasil Model", "ℹ️ Tentang Proyek"])

# ---------------------------------------------------------------- TAB 1
with tab1:
    st.selectbox(
        "Skenario contoh (nilai diambil dari rata-rata data asli pada jam tersebut)",
        list(PRESET.keys()),
        key="skenario",
        on_change=terapkan_preset,
    )
    st.caption("Pilih skenario untuk mengisi nilai otomatis, lalu ubah sendiri bila perlu.")

    def kelompok(judul, daftar, ikon):
        with st.expander(f"{ikon} {judul}", expanded=(judul == "Bacaan sensor murah")):
            kiri, kanan = st.columns(2)
            for i, (label, lo, hi, step, k) in enumerate(daftar):
                with (kiri if i % 2 == 0 else kanan):
                    nilai = float(st.session_state["in_" + k])
                    st.session_state["in_" + k] = min(max(nilai, lo), hi)
                    st.slider(label, lo, hi, step=step, key="in_" + k)

    kelompok("Bacaan sensor murah", SENSOR, "📟")
    kelompok("Gas dari alat referensi", GAS, "🧪")
    kelompok("Cuaca", CUACA, "🌤️")

    with st.expander("🕒 Waktu"):
        a, b, c = st.columns(3)
        with a:
            st.slider("Jam", 0, 23, key="in_Hour")
        with b:
            st.slider("Bulan", 1, 12, key="in_Month")
        with c:
            st.selectbox("Hari", HARI, key="in_Hari")

    if st.button("Prediksi kadar CO", type="primary"):
        s = st.session_state
        data = {k: s["in_" + k] for *_, k in SENSOR + GAS + CUACA}
        data.update({"Hour": s["in_Hour"], "Month": s["in_Month"], "DayOfWeek": HARI.index(s["in_Hari"])})
        x = pd.DataFrame([data])[kolom]  # urutan kolom sama dengan saat pelatihan
        hasil = max(float(model.predict(x)[0]), 0.0)

        if hasil < 0.75 * RATA_CO:
            ket = "di bawah rata-rata dataset"
        elif hasil <= 1.25 * RATA_CO:
            ket = "mendekati rata-rata dataset"
        else:
            ket = "di atas rata-rata dataset"
        posisi = min(hasil / 12.0, 1.0) * 100

        st.markdown(
            f"""
<div class="result">
  <div class="small">PERKIRAAN KADAR CO</div>
  <div class="big">{hasil:.2f} <span class="unit">mg/m³</span></div>
  <div>Hasil ini <b>{ket}</b> (rata-rata dataset ≈ {RATA_CO} mg/m³)</div>
  <div class="gauge"><div class="mark" style="left:{posisi:.1f}%"></div></div>
  <div class="small" style="display:flex;justify-content:space-between;"><span>0</span><span>6</span><span>12 mg/m³</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption("Model: Linear Regression (R² 0,871 pada data uji). Hasil berupa perkiraan, bukan hasil pengukuran.")

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("Perbandingan model regresi")
    st.write("Lima model diuji pada 1.535 data terakhir (pembagian berdasarkan waktu).")
    tabel = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Ridge", "Gradient Boosting", "Random Forest (tuning)", "Random Forest", "Decision Tree"],
            "MAE": [0.319, 0.320, 0.399, 0.376, 0.382, 0.501],
            "RMSE": [0.484, 0.484, 0.595, 0.601, 0.604, 0.872],
            "R²": [0.871, 0.871, 0.804, 0.800, 0.798, 0.580],
        }
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model terbaik", "Linear Regression")
    c2.metric("R² terbaik", "0,871")
    c3.metric("MAE terbaik", "0,319")
    st.bar_chart(tabel.set_index("Model")["R²"])
    st.dataframe(tabel, hide_index=True)
    st.info(
        "MAE dan RMSE: makin kecil makin baik. R²: makin dekat ke 1 makin baik. "
        "Model linear paling akurat karena hubungan sensor dan CO cukup linear."
    )

# ---------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("Tentang proyek")
    st.markdown(
        """
<div class="card"><b>Tujuan.</b> Memprediksi kadar CO (CO(GT), mg/m³) dari bacaan sensor murah dan data cuaca
dengan metode regresi.</div>
<div class="card"><b>Dataset.</b> Air Quality UCI: 9.357 pengukuran per jam (Maret 2004 – April 2005).
Setelah pembersihan data tersisa 7.674 baris yang memiliki nilai CO.</div>
<div class="card"><b>Alur.</b> Pembersihan data (kode −200 menjadi NaN) → analisis data → perbandingan 5 model →
tuning → evaluasi → aplikasi demo ini.</div>
""",
        unsafe_allow_html=True,
    )
    st.warning(
        "Keterbatasan: model memakai juga fitur dari alat referensi (berakhiran GT), data hanya dari satu lokasi "
        "selama ±1 tahun, dan akurasi menurun pada lonjakan CO yang tinggi. Aplikasi ini adalah demo, bukan alat ukur."
    )
    st.markdown("**Tim:** Suci Ramadhani (2417020071) · Nadia Putri Pohan (2417020084) · Fauzan Azim (2417020108)")

st.divider()
st.caption("UAS Big Data dan Machine Learning – Kelas VA · Kelompok 8")
