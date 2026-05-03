import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AeroSense", page_icon="✈️", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');
:root{--bg:#0a0c10;--surface:#111318;--panel:#161a22;--border:#1f2430;
      --accent:#00d4ff;--accent2:#ff6b35;--green:#00ff88;--text:#e2e8f0;--muted:#64748b;}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Mono',monospace!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.hero h1{font-family:'Syne',sans-serif!important;font-size:2.8rem!important;font-weight:800!important;
  background:linear-gradient(135deg,#00d4ff,#ff6b35);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0!important;}
.hero p{color:var(--muted)!important;font-size:.85rem;margin:.3rem 0 1.2rem;}
.mrow{display:flex;gap:1rem;margin:1rem 0;flex-wrap:wrap;}
.mcard{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:1rem 1.3rem;flex:1;min-width:140px;position:relative;overflow:hidden;}
.mcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),transparent);}
.mlabel{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin-bottom:.3rem;}
.mvalue{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:700;color:var(--accent);}
.msub{font-size:.7rem;color:var(--muted);margin-top:.15rem;}
.stitle{font-family:'Syne',sans-serif!important;font-size:.9rem!important;font-weight:700!important;
  color:var(--text)!important;text-transform:uppercase;letter-spacing:3px;
  margin:1.8rem 0 .8rem!important;padding-bottom:.4rem;border-bottom:1px solid var(--border);}
.ibox{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--accent);
  border-radius:8px;padding:.9rem 1.1rem;margin:.8rem 0;font-size:.82rem;color:var(--muted);line-height:1.7;}
.gstep{display:flex;gap:1rem;align-items:flex-start;padding:.9rem;
  background:var(--panel);border:1px solid var(--border);border-radius:10px;margin-bottom:.7rem;}
.snum{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:var(--accent);min-width:2rem;line-height:1;}
.scontent strong{color:var(--text);font-size:.88rem;display:block;margin-bottom:.15rem;}
.scontent span{color:var(--muted);font-size:.78rem;}
.badge{display:inline-block;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);
  color:var(--accent);border-radius:4px;padding:2px 7px;font-size:.68rem;margin:2px;letter-spacing:1px;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;
  font-family:'DM Mono',monospace!important;font-size:.78rem!important;border:none!important;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}
button[kind="primary"]{background:var(--accent)!important;color:#000!important;font-family:'Syne',sans-serif!important;font-weight:700!important;}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
DROP   = ['s1','s5','s6','s10','s16','s18','s19']
USEFUL = [f's{i}' for i in range(1,22) if f's{i}' not in DROP]
FEATS  = ['op1','op2','op3'] + USEFUL
SINFO  = {
    's2':'Fan Inlet Temp','s3':'LPC Outlet Temp','s4':'HPC Outlet Temp',
    's7':'HPC Outlet Pressure','s8':'Physical Fan Speed','s9':'Physical Core Speed',
    's11':'HPC Static Pressure','s12':'Fuel Flow Ratio','s13':'Corrected Fan Speed',
    's14':'Corrected Core Speed','s15':'Bypass Ratio','s17':'Bleed Enthalpy',
    's20':'HP Turbine Speed','s21':'LP Turbine Speed',
}

plt.rcParams.update({
    'figure.facecolor':'#111318','axes.facecolor':'#161a22','axes.edgecolor':'#1f2430',
    'axes.labelcolor':'#64748b','xtick.color':'#64748b','ytick.color':'#64748b',
    'text.color':'#e2e8f0','grid.color':'#1f2430','grid.linewidth':.5,
    'font.family':'monospace','axes.spines.top':False,'axes.spines.right':False,
})

# ── HELPERS ──────────────────────────────────────────────────────────────────
def compute_rul(df, cap):
    mc = df.groupby('unit')['cycle'].max().reset_index()
    mc.columns = ['unit','max_cycle']
    df = df.merge(mc, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL_clipped'] = df['RUL'].clip(upper=cap)
    df.drop(columns=['max_cycle'], inplace=True)
    return df

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Syne;font-size:1.1rem;font-weight:800;color:#00d4ff;letter-spacing:2px;">✈ AEROSENSE</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.65rem;color:#64748b;letter-spacing:1px;">ENGINE HEALTH MONITOR</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**📂 Upload Data**")
    train_f = st.file_uploader("Train CSV", type="csv", key="tr")
    test_f  = st.file_uploader("Test CSV",  type="csv", key="te")
    rul_f   = st.file_uploader("RUL CSV",   type="csv", key="ru")
    st.divider()
    st.markdown("**⚙️ Model Config**")
    rul_cap = st.slider("RUL Cap",       50, 200, 125, 25)
    n_est   = st.slider("N Estimators", 100, 500, 300, 50)
    max_d   = st.slider("Max Depth",      3,  10,   6)
    lr      = st.select_slider("Learning Rate", [0.01,0.03,0.05,0.1,0.2], value=0.05)
    st.divider()
    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>AeroSense</h1>
  <p>Turbofan Engine Degradation &nbsp;·&nbsp; Remaining Useful Life Prediction &nbsp;·&nbsp; NASA C-MAPSS</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📖  PANDUAN","📊  EDA","🎯  PREDIKSI RUL","🔬  FEATURE IMPORTANCE"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PANDUAN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="stitle">Cara Pakai</p>', unsafe_allow_html=True)
    for num, title, desc in [
        ("1","Download Dataset NASA C-MAPSS","Ekstrak ZIP dari NASA Prognostics Repository. Butuh: train_FD001.csv, test_FD001.csv, RUL_FD001.csv"),
        ("2","Upload File di Sidebar","Upload 3 file CSV secara berurutan: Train → Test → RUL"),
        ("3","Atur Hyperparameter (opsional)","Ubah RUL Cap, N Estimators, Max Depth, Learning Rate di sidebar"),
        ("4","Klik Run Analysis","Pipeline otomatis: EDA → Training → Prediksi → Evaluasi"),
        ("5","Eksplorasi Hasil","Tab EDA untuk visualisasi sensor, Prediksi untuk hasil RUL, Feature Importance untuk insight model"),
    ]:
        st.markdown(f"""
        <div class="gstep">
          <div class="snum">{num}</div>
          <div class="scontent"><strong>{title}</strong><span>{desc}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="stitle">Tentang Dataset C-MAPSS</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="ibox"><strong style="color:#00d4ff">Apa itu C-MAPSS?</strong><br>
        Commercial Modular Aero-Propulsion System Simulation — software NASA yang mensimulasikan mesin turbofan
        komersial (tipe yang dipakai Boeing/Airbus). Data berisi pembacaan 21 sensor dari ratusan mesin,
        dari awal operasi sampai failure.</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="ibox"><strong style="color:#ff6b35">Struktur Kolom</strong><br>
        <span class="badge">unit</span> ID mesin &nbsp;
        <span class="badge">cycle</span> siklus ≈ 1 penerbangan<br>
        <span class="badge">op1–op3</span> kondisi terbang (Mach, altitude, throttle)<br>
        <span class="badge">s1–s21</span> 21 sensor (suhu, tekanan, kecepatan, dll)</div>""", unsafe_allow_html=True)

    st.markdown('<p class="stitle">Sensor Informatif</p>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame([{"Sensor":k,"Deskripsi":v} for k,v in SINFO.items()]),
        use_container_width=True, hide_index=True
    )
    st.markdown("""<div class="ibox" style="border-left-color:#ff6b35">
    <strong style="color:#ff6b35">⚠️ Sensor yang dibuang:</strong>
    s1, s5, s6, s10, s16, s18, s19 — nilainya konstan atau pure noise, tidak memberi info degradasi mesin.</div>""",
    unsafe_allow_html=True)

# ── RUN PIPELINE ─────────────────────────────────────────────────────────────
if run_btn:
    if not train_f or not test_f or not rul_f:
        st.error("❌ Upload Train, Test, dan RUL CSV dulu di sidebar!")
    else:
        with st.spinner("Loading data..."):
            train_df = pd.read_csv(train_f)
            test_df  = pd.read_csv(test_f)
            rul_df   = pd.read_csv(rul_f)
            train_df = compute_rul(train_df, rul_cap)

        with st.spinner("Training XGBoost..."):
            scaler = MinMaxScaler()
            X_tr = scaler.fit_transform(train_df[FEATS])
            y_tr = train_df['RUL_clipped'].values
            last = test_df.groupby('unit').last().reset_index()
            X_te = scaler.transform(last[FEATS])
            y_true = rul_df.iloc[:,0].values

            model = xgb.XGBRegressor(
                n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = 1 - np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true))**2)

        st.session_state['res'] = dict(
            train_df=train_df, test_df=test_df,
            y_pred=y_pred, y_true=y_true,
            rmse=rmse, mae=mae, r2=r2, model=model
        )
        st.success("✅ Selesai! Lihat tab EDA, Prediksi, dan Feature Importance.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if 'res' not in st.session_state:
        st.markdown('<div class="ibox">Upload data & klik <strong>Run Analysis</strong> di sidebar dulu.</div>', unsafe_allow_html=True)
    else:
        r = st.session_state['res']
        tr = r['train_df']
        n_eng = tr['unit'].nunique()
        avg_l = tr.groupby('unit')['cycle'].max().mean()
        max_l = tr.groupby('unit')['cycle'].max().max()

        st.markdown(f"""
        <div class="mrow">
          <div class="mcard"><div class="mlabel">Total Mesin</div><div class="mvalue">{n_eng}</div><div class="msub">training engines</div></div>
          <div class="mcard"><div class="mlabel">Rata-rata Umur</div><div class="mvalue">{avg_l:.0f}</div><div class="msub">cycle rata-rata</div></div>
          <div class="mcard"><div class="mlabel">Umur Terpanjang</div><div class="mvalue">{max_l:.0f}</div><div class="msub">cycle tertinggi</div></div>
          <div class="mcard"><div class="mlabel">Total Records</div><div class="mvalue">{len(tr):,}</div><div class="msub">baris data</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<p class="stitle">Distribusi Umur Mesin</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,3.5))
        cc = tr.groupby('unit')['cycle'].max()
        ax.hist(cc, bins=20, color='#00d4ff', alpha=0.85, edgecolor='#0a0c10')
        ax.axvline(cc.mean(), color='#ff6b35', linestyle='--', linewidth=1.5, label=f'Mean = {cc.mean():.0f}')
        ax.set_xlabel('Total Cycle sampai Failure'); ax.set_ylabel('Jumlah Mesin')
        ax.legend(fontsize=9)
        st.pyplot(fig); plt.close()

        st.markdown('<p class="stitle">Tren Sensor per Engine</p>', unsafe_allow_html=True)
        sel_eng = st.selectbox("Pilih Engine", sorted(tr['unit'].unique().tolist()), index=0)
        eng_df  = tr[tr['unit']==sel_eng]
        display = list(SINFO.keys())[:9]
        fig, axes = plt.subplots(3,3,figsize=(14,8))
        for i, s in enumerate(display):
            if s in eng_df.columns:
                axes.flatten()[i].plot(eng_df['cycle'], eng_df[s], color='#00d4ff', linewidth=1.2)
                axes.flatten()[i].set_title(SINFO.get(s,s), fontsize=8, color='#e2e8f0')
                axes.flatten()[i].set_xlabel('Cycle', fontsize=7)
        plt.suptitle(f'Engine #{sel_eng} — Sensor Trends', fontsize=11, color='#e2e8f0', y=1.01)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<p class="stitle">Korelasi Antar Sensor</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12,7))
        corr = tr[USEFUL].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                    linewidths=0.3, linecolor='#0a0c10', ax=ax, cbar_kws={'shrink':.8})
        ax.set_title('Sensor Correlation Matrix', color='#e2e8f0', pad=10)
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if 'res' not in st.session_state:
        st.markdown('<div class="ibox">Upload data & klik <strong>Run Analysis</strong> di sidebar dulu.</div>', unsafe_allow_html=True)
    else:
        r = st.session_state['res']
        y_pred, y_true = r['y_pred'], r['y_true']
        rmse, mae, r2  = r['rmse'], r['mae'], r['r2']

        st.markdown(f"""
        <div class="mrow">
          <div class="mcard"><div class="mlabel">RMSE</div><div class="mvalue" style="color:#ff6b35">{rmse:.1f}</div><div class="msub">Root Mean Squared Error</div></div>
          <div class="mcard"><div class="mlabel">MAE</div><div class="mvalue" style="color:#00ff88">{mae:.1f}</div><div class="msub">cycle meleset rata-rata</div></div>
          <div class="mcard"><div class="mlabel">R² Score</div><div class="mvalue" style="color:#00d4ff">{r2:.3f}</div><div class="msub">1.0 = perfect</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<p class="stitle">Prediksi vs Aktual RUL</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14,4.5))
        x = np.arange(len(y_true))
        ax.fill_between(x, y_true, alpha=0.12, color='#00d4ff')
        ax.plot(x, y_true, color='#00d4ff', linewidth=1.5, label='RUL Aktual')
        ax.plot(x, y_pred, color='#ff6b35', linewidth=1.2, alpha=0.85, label='RUL Prediksi')
        ax.set_xlabel('Engine Index'); ax.set_ylabel('RUL (cycle)')
        ax.legend(fontsize=9); st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="stitle">Scatter Prediksi vs Aktual</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(y_true, y_pred, alpha=0.5, color='#00d4ff', edgecolors='#0a0c10', s=40)
            mv = max(y_true.max(), y_pred.max()) + 10
            ax.plot([0,mv],[0,mv],'--', color='#ff6b35', linewidth=1.5, label='Perfect')
            ax.set_xlabel('RUL Aktual'); ax.set_ylabel('RUL Prediksi')
            ax.legend(fontsize=9); st.pyplot(fig); plt.close()
        with c2:
            st.markdown('<p class="stitle">Distribusi Error</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6,6))
            err = y_pred - y_true
            ax.hist(err, bins=25, color='#00ff88', alpha=0.85, edgecolor='#0a0c10')
            ax.axvline(0, color='#ff6b35', linestyle='--', linewidth=1.5)
            ax.set_xlabel('Prediction Error (pred - actual)'); ax.set_ylabel('Frekuensi')
            st.pyplot(fig); plt.close()

        st.markdown('<p class="stitle">Sample Hasil Prediksi (20 Engine Pertama)</p>', unsafe_allow_html=True)
        res_df = pd.DataFrame({
            'Engine': np.arange(1, len(y_true)+1),
            'RUL Aktual': y_true.astype(int),
            'RUL Prediksi': y_pred.round(1),
            'Error': (y_pred - y_true).round(1),
        })
        res_df['Status'] = res_df['Error'].apply(
            lambda e: '🟢 OK' if abs(e)<=20 else ('🟡 Warning' if abs(e)<=40 else '🔴 Jauh')
        )
        st.dataframe(res_df.head(20), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if 'res' not in st.session_state:
        st.markdown('<div class="ibox">Upload data & klik <strong>Run Analysis</strong> di sidebar dulu.</div>', unsafe_allow_html=True)
    else:
        model = st.session_state['res']['model']
        imp = pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=True)

        st.markdown('<p class="stitle">Feature Importance — XGBoost</p>', unsafe_allow_html=True)
        st.markdown("""<div class="ibox">
        Semakin tinggi importance score, semakin besar kontribusi sensor tersebut dalam memprediksi RUL.
        Sensor dengan score rendah bisa dipertimbangkan untuk dibuang (feature selection).</div>""",
        unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10,8))
        colors = ['#00d4ff' if v > imp.median() else '#2a3040' for v in imp]
        bars = ax.barh(imp.index, imp.values, color=colors, edgecolor='#0a0c10', height=0.7)
        ax.axvline(imp.median(), color='#ff6b35', linestyle='--', linewidth=1, alpha=0.7, label='Median')
        ax.set_xlabel('Importance Score')
        ax.set_title('XGBoost Feature Importance', color='#e2e8f0', pad=12)
        ax.legend(fontsize=9)
        for bar, val in zip(bars, imp.values):
            ax.text(val+.001, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=7, color='#64748b')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<p class="stitle">Top 5 Sensor Terpenting</p>', unsafe_allow_html=True)
        top5 = imp.sort_values(ascending=False).head(5).reset_index()
        top5.columns = ['Feature','Importance Score']
        top5['Deskripsi'] = top5['Feature'].map(lambda x: SINFO.get(x,'Operational Setting'))
        top5['Importance Score'] = top5['Importance Score'].round(4)
        st.dataframe(top5, use_container_width=True, hide_index=True)
