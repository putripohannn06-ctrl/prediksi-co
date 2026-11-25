import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Kadar CO", page_icon="🌫️", layout="centered")

HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
RATA_CO = 2.15  # rata-rata CO(GT) pada dataset (mg/m³)

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

# Nilai rata-rata/tengah dari data asli pada jam tertentu
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

MODEL_TABEL = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Ridge", "Gradient Boosting", "Random Forest (tuning)", "Random Forest", "Decision Tree"],
        "MAE": [0.319, 0.320, 0.399, 0.376, 0.382, 0.501],
        "RMSE": [0.484, 0.484, 0.595, 0.601, 0.604, 0.872],
        "R²": [0.871, 0.871, 0.804, 0.800, 0.798, 0.580],
    }
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {font-family: 'Inter', sans-serif;}
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 920px;}
footer {visibility: hidden;}
.hero {background: radial-gradient(circle at 85% 20%, #166534 0%, rgba(22,101,52,0) 45%), linear-gradient(135deg, #0B0F14 0%, #12361F 100%);
       color: #fff; padding: 30px 32px 26px 32px; border-radius: 22px; margin-bottom: 18px;
       box-shadow: 0 10px 30px rgba(11,15,20,.18);}
.hero .badge {display:inline-block; background:#22C55E; color:#0B0F14; font-weight:800; letter-spacing:.04em;
       padding: 4px 14px; border-radius: 999px; font-size: .72rem; margin-bottom: 14px;}
.hero h1 {margin: 0 0 8px 0; padding: 0; font-size: 2.3rem; font-weight: 800; color: #fff; line-height: 1.15;}
.hero p {margin: 0 0 18px 0; color: #BBF7D0; font-size: .98rem;}
.chips {display:flex; flex-wrap:wrap; gap:10px;}
.chip {background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18); border-radius: 12px;
       padding: 8px 14px; font-size: .82rem; color:#E7F8EC;}
.chip b {color:#fff; font-size: 1.05rem; display:block;}
.steps {display:flex; flex-wrap:wrap; gap:10px; margin: 4px 0 14px 0;}
.step {flex:1 1 200px; background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:12px 14px; font-size:.88rem; color:#334155;}
.step .n {display:inline-block; width:24px; height:24px; line-height:24px; text-align:center; border-radius:50%;
       background:#16A34A; color:#fff; font-weight:700; font-size:.8rem; margin-right:8px;}
.card {background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:16px 20px; margin-bottom:12px;
       box-shadow: 0 1px 2px rgba(15,23,42,.04);}
.card h4 {margin:0 0 6px 0; font-size:1rem; color:#14532D;}
.card p {margin:0; color:#475569; font-size:.92rem; line-height:1.55;}
.result {background:#fff; border:2px solid #22C55E; border-radius:22px; padding:26px 22px 20px 22px; text-align:center;
       box-shadow: 0 12px 30px rgba(34,197,94,.15); margin-top: 6px;}
.result .lbl {color:#64748B; font-size:.78rem; font-weight:700; letter-spacing:.12em;}
.result .big {font-size: 3.6rem; font-weight: 800; color:#14532D; line-height:1.1; margin: 6px 0;}
.result .unit {color:#64748B; font-size:1.05rem; font-weight:600;}
.tag {display:inline-block; padding:5px 14px; border-radius:999px; font-size:.85rem; font-weight:700; margin: 4px 0 2px 0;}
.tag.low {background:#DCFCE7; color:#166534;}
.tag.mid {background:#FEF3C7; color:#92400E;}
.tag.high {background:#FEE2E2; color:#991B1B;}
.gauge {position:relative; height:14px; border-radius:999px; margin: 22px 8px 6px 8px;
       background: linear-gradient(90deg,#86EFAC 0%,#FDE68A 45%,#FCA5A5 100%);}
.mark {position:absolute; top:-7px; width:6px; height:28px; background:#0B0F14; border-radius:3px; box-shadow:0 0 0 3px #fff;}
.ticks {display:flex; justify-content:space-between; color:#64748B; font-size:.78rem; margin: 0 4px;}
.mini {display:flex; flex-wrap:wrap; gap:10px; margin-top:16px;}
.mini div {flex:1 1 140px; background:#F8FAFC; border-radius:12px; padding:10px; font-size:.8rem; color:#64748B;}
.mini div b {display:block; font-size:1.15rem; color:#0B0F14;}
.kpis {display:flex; flex-wrap:wrap; gap:12px; margin: 6px 0 16px 0;}
.kpi {flex:1 1 200px; background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:14px 18px;}
.kpi span {color:#64748B; font-size:.8rem; font-weight:600;}
.kpi b {display:block; font-size:1.55rem; color:#14532D; font-weight:800; margin-top:2px;}
.bars {background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:16px 20px; margin-bottom:14px;}
.bars .row {display:flex; align-items:center; gap:12px; margin: 9px 0; font-size:.9rem; color:#334155;}
.bars .name {flex:0 0 190px; font-weight:600;}
.bars .track {flex:1; background:#EEF2F6; border-radius:999px; height:18px; overflow:hidden;}
.bars .fill {height:100%; border-radius:999px; background:#94A3B8;}
.bars .fill.best {background: linear-gradient(90deg,#16A34A,#22C55E);}
.bars .val {flex:0 0 52px; text-align:right; font-weight:700;}
.team {display:flex; flex-wrap:wrap; gap:10px;}
.person {flex:1 1 230px; background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:12px 16px; font-size:.9rem;}
.person b {display:block; color:#0B0F14;}
.person span {color:#64748B; font-size:.82rem;}
.note {background:#FFFBEB; border:1px solid #FDE68A; color:#92400E; border-radius:14px; padding:12px 16px; font-size:.88rem; margin: 6px 0 12px 0;}
.foot {text-align:center; color:#94A3B8; font-size:.8rem; margin-top: 22px;}
/* warna utama (cadangan bila tema belum terbaca) */
button[data-baseweb="tab"][aria-selected="true"] {color:#15803D !important; font-weight:700;}
div[data-baseweb="tab-highlight"] {background-color:#16A34A !important;}
div[data-testid="stSlider"] [role="slider"] {background-color:#16A34A !important;}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {background:#16A34A !important; border:none !important;
       color:#fff !important; font-weight:700; border-radius:12px; padding:.7rem 1rem;}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {background:#15803D !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """<div class="hero">
<span class="badge">DEMO · UAS BIG DATA DAN MACHINE LEARNING</span>
<h1>Prediksi Kadar CO</h1>
<p>Memperkirakan CO(GT) dari bacaan sensor dan cuaca · Dataset Air Quality UCI · Kelompok 8</p>
<div class="chips">
<div class="chip"><b>0,871</b>R² model terbaik</div>
<div class="chip"><b>7.674</b>data bernilai CO</div>
<div class="chip"><b>14</b>fitur masukan</div>
<div class="chip"><b>5</b>model dibandingkan</div>
</div></div>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def muat_model():
    return joblib.load("model_co.joblib"), joblib.load("kolom_fitur.joblib")


try:
    model, kolom = muat_model()
except Exception as e:  # file tidak ditemukan / versi library berbeda
    st.error(f"Model gagal dimuat: {e}")
    st.stop()


def terapkan_preset():
    for k, v in PRESET[st.session_state["skenario"]].items():
        st.session_state["in_" + k] = v


if "in_Hour" not in st.session_state:
    for k, v in PRESET["Nilai tengah data"].items():
        st.session_state["in_" + k] = v

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Hasil Model", "ℹ️ Tentang Proyek"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    st.markdown(
        """<div class="steps">
<div class="step"><span class="n">1</span>Pilih skenario contoh</div>
<div class="step"><span class="n">2</span>Atur bacaan sensor dan cuaca</div>
<div class="step"><span class="n">3</span>Klik <b>Prediksi kadar CO</b></div>
</div>""",
        unsafe_allow_html=True,
    )
    st.selectbox(
        "Skenario contoh (nilai dari rata-rata data asli pada jam tersebut)",
        list(PRESET.keys()),
        key="skenario",
        on_change=terapkan_preset,
    )

    def kelompok(judul, daftar, ikon, terbuka=False):
        with st.expander(f"{ikon} {judul}", expanded=terbuka):
            kiri, kanan = st.columns(2)
            for i, (label, lo, hi, step, k) in enumerate(daftar):
                with (kiri if i % 2 == 0 else kanan):
                    st.session_state["in_" + k] = min(max(float(st.session_state["in_" + k]), lo), hi)
                    st.slider(label, lo, hi, step=step, key="in_" + k)

    kelompok("Bacaan sensor murah", SENSOR, "📟", terbuka=True)
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
            kelas, ket = "low", "Di bawah rata-rata dataset"
        elif hasil <= 1.25 * RATA_CO:
            kelas, ket = "mid", "Mendekati rata-rata dataset"
        else:
            kelas, ket = "high", "Di atas rata-rata dataset"
        posisi = min(hasil / 12.0, 1.0) * 100
        selisih = (hasil / RATA_CO - 1) * 100
        tanda = "+" if selisih >= 0 else "−"

        st.markdown(
            f"""<div class="result">
<div class="lbl">PERKIRAAN KADAR CO</div>
<div class="big">{hasil:.2f} <span class="unit">mg/m³</span></div>
<span class="tag {kelas}">{ket}</span>
<div class="gauge"><div class="mark" style="left:calc({posisi:.1f}% - 3px)"></div></div>
<div class="ticks"><span>0</span><span>3</span><span>6</span><span>9</span><span>12 mg/m³</span></div>
<div class="mini">
<div>Hasil prediksi<b>{hasil:.2f}</b></div>
<div>Rata-rata dataset<b>{RATA_CO:.2f}</b></div>
<div>Selisih dari rata-rata<b>{tanda}{abs(selisih):.0f}%</b></div>
</div></div>""",
            unsafe_allow_html=True,
        )
        st.caption("Model: Linear Regression (R² 0,871 pada data uji). Hasil berupa perkiraan, bukan hasil pengukuran.")
        with st.expander("Lihat data masukan yang dipakai model"):
            st.dataframe(x.T.rename(columns={0: "Nilai"}))

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.subheader("Perbandingan model regresi")
    st.write("Lima model diuji pada 1.535 data terakhir (pembagian berdasarkan waktu).")
    st.markdown(
        """<div class="kpis">
<div class="kpi"><span>Model terbaik</span><b>Linear Regression</b></div>
<div class="kpi"><span>R² terbaik</span><b>0,871</b></div>
<div class="kpi"><span>MAE terbaik</span><b>0,319</b></div>
</div>""",
        unsafe_allow_html=True,
    )
    urut = MODEL_TABEL.sort_values("R²", ascending=False)
    baris = ""
    for _, r in urut.iterrows():
        terbaik = " best" if r["R²"] >= 0.871 else ""
        baris += (
            f'<div class="row"><div class="name">{r["Model"]}</div>'
            f'<div class="track"><div class="fill{terbaik}" style="width:{r["R²"] * 100:.1f}%"></div></div>'
            f'<div class="val">{r["R²"]:.3f}</div></div>'
        )
    st.markdown(f'<div class="bars"><b>Skor R² (makin besar makin baik)</b>{baris}</div>', unsafe_allow_html=True)
    st.dataframe(MODEL_TABEL, hide_index=True)
    st.markdown(
        """<div class="card"><h4>Cara membaca</h4>
<p>MAE dan RMSE: makin kecil makin baik. R²: makin dekat ke 1 makin baik.
Model linear paling akurat karena hubungan antara bacaan sensor dan CO cukup linear,
sehingga model yang lebih rumit tidak otomatis lebih baik.</p></div>""",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ TAB 3
with tab3:
    st.subheader("Tentang proyek")
    st.markdown(
        """<div class="card"><h4>Tujuan</h4>
<p>Memprediksi kadar CO (CO(GT), mg/m³) dari bacaan sensor murah dan data cuaca dengan metode regresi.</p></div>
<div class="card"><h4>Dataset</h4>
<p>Air Quality UCI: 9.357 pengukuran per jam (Maret 2004 – April 2005). Setelah pembersihan data,
tersisa 7.674 baris yang memiliki nilai CO.</p></div>
<div class="steps">
<div class="step"><span class="n">1</span>Bersihkan data (kode −200 → NaN)</div>
<div class="step"><span class="n">2</span>Analisis data (EDA)</div>
<div class="step"><span class="n">3</span>Bandingkan 5 model + tuning</div>
<div class="step"><span class="n">4</span>Evaluasi dan aplikasi demo</div>
</div>
<div class="note"><b>Keterbatasan.</b> Model memakai juga fitur dari alat referensi (berakhiran GT), data hanya dari satu lokasi
selama ±1 tahun, dan akurasi menurun pada lonjakan CO yang tinggi. Aplikasi ini adalah demo, bukan alat ukur.</div>
<div class="team">
<div class="person"><b>Suci Ramadhani</b><span>2417020071</span></div>
<div class="person"><b>Nadia Putri Pohan</b><span>2417020084</span></div>
<div class="person"><b>Fauzan Azim</b><span>2417020108</span></div>
</div>""",
        unsafe_allow_html=True,
    )

st.markdown('<div class="foot">UAS Big Data dan Machine Learning – Kelas VA · Kelompok 8</div>', unsafe_allow_html=True)
