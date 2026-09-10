# Skenario Surabaya

Skenario ini memprediksi tinggi air perkotaan dalam **sentimeter**, dengan target awal **Hang Tuah** (`ketinggian_lokasi_1_hang_tuah`). Data berinterval 30 menit; horizon prediksi +1 sampai +5 jam. Nama teknis `urban` pada modul dan model merujuk pada Surabaya.

## 1. Data dan EDA

Dataset ada di `data/surabaya/ketinggian_30menit_wide.csv`, dengan kolom waktu `timestamp`. Konfigurasi mencakup Hang Tuah, Kalibokor, Rumah Pompa Pucang, dan PLN Menur, masing-masing dengan sinyal tinggi air dan curah hujan serta komponen sensor A/B.

Mulai dari `research/surabaya/01_eda_surabaya.ipynb`. Notebook membahas coverage, missingness, outlier, korelasi, pola temporal, dan perbandingan dengan persistence. Jalankan sel berurutan dari root repository atau folder notebook. Output tersimpan adalah hasil eksekusi sebelumnya dan tetap dipertahankan.

## 2. Preprocessing dan fitur

- `src/dhompo/data/urban_loader.py`: kanonisasi sensor A/B, pemeriksaan coverage, quality flag, pembersihan target, dan forward fill terbatas.
- `src/dhompo/data/urban_features.py`: nilai terakhir, lag, rolling statistics, perubahan nilai, fitur waktu, dan quality flag.
- `configs/surabaya/urban_water_level.yaml`: target, pemetaan sensor, fitur, cleaning, split, dan direktori output.

Default konfigurasi memakai minimal history 24 baris, forward fill maksimal 12 langkah, serta split temporal 80% train dan 20% test setelah penyelarasan fitur dan target. Nilai nol pada target diperlakukan sebagai outlier sesuai konfigurasi cleaning. Banyak sensor memiliki coverage rendah; jangan menganggap semua kolom selalu dapat digunakan sebagai fitur.

## 3. Variasi eksperimen

| Konfigurasi | Mode target | Model output |
|---|---|---|
| `configs/surabaya/urban_water_level.yaml` | `persistence_residual` (default) | `models/surabaya/urban/` |
| `configs/surabaya/urban_water_level_delta.yaml` | `delta` | `models/surabaya/urban_delta/` |

Kedua mode memodelkan perubahan relatif terhadap nilai saat ini, lalu menambahkannya kembali untuk menghasilkan prediksi tinggi air. Kode juga mendukung mode `level` untuk target tinggi air langsung. Pertahankan konfigurasi dan metadata setiap run agar hasil eksperimen dapat ditelusuri.

## 4. Menjalankan pipeline

Dari root repository setelah instalasi development:

```powershell
# Memeriksa data, fitur, dan split tanpa melatih model
python training/surabaya/train_urban_sklearn.py --dry-run

# Training default residual terhadap persistence
python training/surabaya/train_urban_sklearn.py --models ridge

# Menjalankan varian delta dengan direktori output terpisah
python training/surabaya/train_urban_sklearn.py --config configs/surabaya/urban_water_level_delta.yaml --models ridge
```

Training memakai hyperparameter bersama di `configs/shared/sklearn_model.yaml`. Opsi `--models` juga menerima daftar seperti `gradient_boosting,xgboost`; gunakan `--help` untuk pilihan argumen. Training menulis model, scaler, dan `training_metadata.json` ke direktori output konfigurasi, sehingga dapat menimpa run sebelumnya pada direktori yang sama.

Membuat laporan dari artefak yang sudah tersedia:

```powershell
python training/surabaya/report_urban_predictions.py

python training/surabaya/report_urban_predictions.py --config configs/surabaya/urban_water_level_delta.yaml --model-dir models/surabaya/urban_delta --output reports/surabaya/tables/surabaya_delta.xlsx
```

Laporan default berada di `reports/surabaya/tables/xls_17_surabaya_residual_vs_persistence_ringkas.xlsx`. Isinya mencakup ringkasan metrik serta prediksi versus ground truth. Persistence dan trend persistence disertakan sebagai pembanding kecuali opsi `--no-persistence-baselines` digunakan.

## 5. Prediksi melalui Python

Model, `standard_scaler_global.pkl`, dan `training_metadata.json` harus berasal dari run yang sama. Model hasil training tidak dilacak Git; metadata saja belum cukup untuk prediksi.

Contoh berikut memprediksi dari waktu terakhir dengan target valid pada dataset historis lokal. Bagian akhir CSV dapat memiliki target kosong setelah batas forward fill; pemilihan ini ditujukan untuk contoh historis, bukan untuk menyamarkan sensor yang sedang tidak tersedia pada layanan real-time.

```python
from dhompo.data.urban_loader import preprocess_urban_wide_data
from dhompo.serving.urban_file_predictor import UrbanFilePredictor

data = preprocess_urban_wide_data()
predictor = UrbanFilePredictor()
last_valid = data.values[data.target_column].last_valid_index()
if last_valid is None:
    raise ValueError("Dataset tidak memiliki target yang valid")
result = predictor.predict_from_history(
    data.values.loc[:last_valid], data.quality_flags.loc[:last_valid]
)
print(result.predictions)  # h1–h5 dalam sentimeter
```

Ini adalah penggunaan predictor Python. FastAPI yang ada saat ini melayani Dhompo dan memakai schema stasiun serta satuan berbeda.

## 6. Membaca hasil

Gunakan NSE, RMSE, MAE, R², dan PBIAS dari metrik bersama `training/evaluate.py`. RMSE/MAE Surabaya dalam sentimeter. Selalu bandingkan model dengan persistence pada periode uji yang sama; autokorelasi target yang kuat dapat membuat baseline persistence sulit dikalahkan.

Periksa metadata untuk fitur terpilih, coverage sensor, cleaning, rentang waktu, jumlah baris train/test, mode target, serta model terbaik tiap horizon. Jangan membandingkan angka RMSE mentah Surabaya dengan Dhompo karena satuannya berbeda.

Artefak lokal lama dapat mengeluarkan peringatan perbedaan versi scikit-learn atau XGBoost ketika dimuat dalam environment baru. Gunakan versi library dari run asal untuk reproduksi ilmiah; penataan folder tidak mengonversi atau melatih ulang artefak tersebut.
