# Training dan Evaluasi

- `dhompo/`: validasi data, training baseline/deep learning/adaptive, tuning, eksperimen, serta pembuatan laporan Dhompo.
- `surabaya/`: training baseline Surabaya dan laporan prediksi versus ground truth.
- `evaluate.py`: metrik yang digunakan bersama oleh kedua skenario.

Baseline 15 stasiun Dhompo untuk horizon +1 sampai +6 jam:

```powershell
python training/dhompo/train_multistation.py --dry-run
python training/dhompo/train_multistation.py
```

Konfigurasi ada di `configs/dhompo/multistation_training.yaml`. Hasil metrik
ditulis ke `reports/dhompo/tables/` dan artefak lokal ke
`models/sklearn/multistation/`.

Simulator skenario 15 stasiun (kalibrasi hujan/routing dan dataset sintetis):

```powershell
python scripts/build_synthetic.py --calibrate-only
python scripts/build_synthetic.py
```

Konfigurasi dan parameter hasil kalibrasi berada di
`configs/dhompo/simulator*.yaml`; CSV sintetis tetap lokal di
`data/synthetic/`.

Training simulator + korektor residual multi-stasiun:

```powershell
python training/dhompo/train_hybrid.py --dry-run
python training/dhompo/train_hybrid.py
```

Model residual, model direct diagnostik, routing, dan metadata disimpan lokal
di `models/sklearn/hybrid/`.

Evaluasi probabilistik pada test origin non-overlap:

```powershell
python training/dhompo/evaluate_hybrid_ensemble.py
```

Evaluator menulis metrik coverage/CRPS ke `reports/dhompo/tables/` dan dua
figur lokal ke `reports/dhompo/figures/`. Quantile ensemble adalah sebaran
skenario hujan, bukan interval prediksi terkalibrasi.

Perintah lengkap dan urutan pemakaian tersedia dalam [panduan Dhompo](../docs/dhompo/index.md) dan [panduan Surabaya](../docs/surabaya/index.md). Jalankan perintah dari root repository; training dapat menimpa hasil pada direktori output konfigurasi yang sama.
