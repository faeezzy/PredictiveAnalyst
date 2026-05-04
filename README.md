# ✈️ AeroSense — Turbofan Engine RUL Predictor

> Predictive Maintenance system untuk memprediksi **Remaining Useful Life (RUL)** mesin turbofan menggunakan **XGBoost**, berbasis dataset **NASA C-MAPSS**, dideploy sebagai interactive dashboard dengan **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-ff4b4b?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-blue?style=flat-square)

---

## 📌 Tentang Project

Mesin pesawat membutuhkan perawatan yang tepat waktu — terlalu cepat buang biaya, terlalu lambat bisa berbahaya. **Predictive Maintenance** hadir sebagai solusi: memprediksi kapan mesin akan rusak *sebelum* kejadian, berdasarkan data sensor real-time.

Project ini membangun model yang memprediksi **berapa siklus operasi (penerbangan) lagi** yang bisa dilakukan sebuah mesin sebelum mengalami failure — disebut **Remaining Useful Life (RUL)**.

---

## 🗂️ Dataset — NASA C-MAPSS

Dataset berasal dari **NASA Prognostics Center of Excellence**, hasil simulasi mesin turbofan komersial (tipe yang digunakan pada Boeing/Airbus).

| File | Deskripsi |
|---|---|
| `train_FD001.csv` | Data sensor 100 mesin dari awal operasi sampai failure |
| `test_FD001.csv` | Data sensor 100 mesin yang di-cut sebelum failure |
| `RUL_FD001.csv` | Ground truth RUL untuk tiap mesin di data test |

**Struktur kolom:**
- `unit` — ID mesin
- `cycle` — siklus operasi (≈ 1 penerbangan)
- `op1–op3` — kondisi terbang (Mach number, altitude, throttle)
- `s1–s21` — 21 sensor (suhu, tekanan, kecepatan turbin, dll)

> Download dataset: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## 🔧 Pipeline

```
train_FD001.csv
      │
      ▼
[1. EDA] ──────────── visualisasi tren sensor, distribusi umur mesin
      │
      ▼
[2. Feature Engineering] ── hitung label RUL, drop sensor konstan, RUL cap
      │
      ▼
[3. Preprocessing] ──── MinMax normalisasi fitur
      │
      ▼
[4. Training] ────────── XGBoost Regressor
      │
      ▼
[5. Prediksi] ────────── input: test_FD001.csv (last cycle tiap mesin)
      │
      ▼
[6. Evaluasi] ────────── RMSE, MAE, R² vs RUL_FD001.csv
```

---

## 📊 Fitur App

| Tab | Isi |
|---|---|
| 📖 Panduan | Cara pakai app, penjelasan dataset & kolom |
| 📊 EDA | Distribusi umur mesin, tren sensor, heatmap korelasi |
| 🎯 Prediksi RUL | Metrik evaluasi, grafik prediksi vs aktual, tabel hasil |
| 🔬 Feature Importance | Sensor mana yang paling berpengaruh terhadap model |

---

## 🚀 Cara Jalanin Lokal

**1. Clone repo**
```bash
git clone https://github.com/faeezzy/PredictiveAnalyst.git
cd aerosense
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Jalanin app**
```bash
streamlit run app.py
```

**4. Upload data**
Buka browser → upload `train_FD001.csv`, `test_FD001.csv`, `RUL_FD001.csv` lewat sidebar → klik **Run Analysis**

---

## 📦 Dependencies

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

---

## 📁 Struktur Repo

```
aerosense/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
└── README.md               # Dokumentasi ini
```

> ⚠️ File CSV tidak di-push ke repo karena ukurannya besar. Download langsung dari NASA dan upload lewat sidebar app.

---

## 🧠 Model — XGBoost Regressor

XGBoost dipilih karena menggunakan pendekatan **boosting** — setiap pohon keputusan baru memperbaiki kesalahan pohon sebelumnya secara sequential, menghasilkan prediksi yang makin akurat secara bertahap.

**Hyperparameter yang bisa diatur:**

| Parameter | Default | Fungsi |
|---|---|---|
| RUL Cap | 125 | Batas atas RUL — fokuskan model ke fase degradasi |
| N Estimators | 300 | Jumlah pohon keputusan |
| Max Depth | 6 | Kedalaman tiap pohon |
| Learning Rate | 0.05 | Seberapa besar koreksi tiap pohon baru |

**Metrik evaluasi:**

| Metrik | Arti |
|---|---|
| MAE | Rata-rata meleset berapa cycle |
| RMSE | Penalti lebih besar untuk error yang jauh |
| R² Score | Seberapa baik model menjelaskan variasi data (1.0 = perfect) |

---

## 👤 Author

**Fai** — Data Science Student, Telkom University  
*Background: Data Science & Embedded Systems/Electronics*

---

## 📄 Referensi

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. NASA Ames Research Center.
- [NASA C-MAPSS Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
