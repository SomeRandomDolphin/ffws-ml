# Hasil Fase 3 — Simulator + Korektor Residual

## Metode

Model hybrid menggunakan simulator transfer jaringan Fase 2 sebagai backbone.
Untuk setiap origin waktu, simulator menerima 48 observasi terakhir dari 15
stasiun dan memproyeksikan +1 sampai +6 jam. Karena prakiraan hujan tidak
tersedia, evaluasi default memakai asumsi hujan masa depan nol.

Korektor multi-output HistGradientBoosting belajar residual:

```text
residual(h) = level_acuan(h) - level_simulator(h)
prediksi_hybrid(h) = level_simulator(h) + ML_residual(h)
```

Dataset sintetis 10 tahun dipakai untuk menguji stabilitas/rentang simulator,
bukan sebagai label korektor. Karena dataset tersebut dibuat oleh simulator
yang sama, melatih residual padanya hanya menghasilkan target mendekati nol dan
tidak menambahkan informasi independen.

Fitur korektor berisi 184 fitur baseline + 15 output simulator untuk horizon
terkait. Routing dikalibrasi **hanya pada bagian train** setiap segmen. Split
80/20 temporal memakai purge gap 6 jam untuk mencegah overlap target train
dengan periode test.

## Hasil test makro

| Horizon | Persistence NSE | Simulator NSE | Direct ML NSE | Hybrid NSE | Direct RMSE | Hybrid RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| +1 jam | 0.6624 | 0.7386 | **0.8345** | 0.8305 | **0.3226** | 0.3235 |
| +2 jam | 0.5955 | 0.6251 | **0.7848** | 0.7818 | 0.3635 | **0.3627** |
| +3 jam | 0.4908 | 0.5306 | 0.7387 | **0.7438** | 0.3939 | **0.3901** |
| +4 jam | 0.3816 | 0.4377 | 0.6731 | **0.6790** | 0.4376 | **0.4339** |
| +5 jam | 0.2892 | 0.3573 | 0.6125 | **0.6208** | 0.4721 | **0.4676** |
| +6 jam | 0.1755 | 0.2848 | 0.5495 | **0.5590** | 0.5071 | **0.5022** |

Simulator sendiri meningkatkan NSE terhadap persistence di seluruh horizon,
tetapi tidak mengalahkan ML. Hybrid sedikit lebih buruk dari direct ML pada
h1–h2 menurut NSE, lalu unggul makro pada h3–h6. Pada h6, hybrid menaikkan NSE
0,0095 dan menurunkan RMSE sekitar 1,0% dibanding direct ML.

## Analisis per stasiun

- Jumlah stasiun dengan NSE hybrid lebih baik dari direct ML: 6/15 (h1),
  8/15 (h2), 12/15 (h3), 11/15 (h4), 9/15 (h5), dan 11/15 (h6).
- Dhompo membaik tipis pada h3 (NSE 0,9522 → 0,9525) dan h6
  (0,7228 → 0,7243), tetapi memburuk pada h1, h2, h4, dan h5.
- `Purwodadi` memperoleh peningkatan h6 terbesar (ΔNSE +0,0658).
- `Bd. Baong` tetap menjadi kasus tersulit; NSE hybrid h1–h6 sekitar
  0,01–0,11. Simulator tidak dapat sepenuhnya memperbaiki sinyal acuan yang
  nyaris white noise.

Hasil test tidak dipakai untuk memilih model per horizon karena itu akan
menjadikan test sebagai tuning set. Pure hybrid tetap menjadi artefak Fase 3;
kebijakan direct/hybrid per horizon memerlukan validation split terpisah.

## Inferensi skenario

`HybridMultiStationPredictor` menerima 48 baris sejarah dan opsional 12 nilai
hujan 30-menit masa depan. Jika hujan tidak diberikan, predictor memakai asumsi
nol dan menandai `future_rainfall_mode=zero`.

Smoke test pada 48 observasi terakhir 2023 menghasilkan Dhompo h6:

- Skenario hujan nol: 9,9743 m.
- Hujan 2 mm/30 menit selama 6 jam: 10,7234 m.

Nilai ini adalah respons skenario model, bukan prediksi cuaca atau validasi
operasional.

## Artefak

- Metrik lengkap: `reports/dhompo/tables/fase3_hybrid_metrics.csv`.
- Ringkasan makro: `reports/dhompo/tables/fase3_hybrid_summary.csv`.
- Enam korektor: `models/sklearn/hybrid/residual_h1..h6.joblib`.
- Enam model direct diagnostik: `models/sklearn/hybrid/direct_h1..h6.joblib`.
- Routing train-only: `models/sklearn/hybrid/routing_parameters.joblib`.
- Predictor: `src/dhompo/serving/hybrid_predictor.py`.

## Reproduksi

```powershell
python training/dhompo/train_hybrid.py --dry-run
python training/dhompo/train_hybrid.py
python -m pytest -q
```

## Kesimpulan

Hipotesis hybrid diterima secara terbatas: backbone fisik-terstruktur memberi
manfaat kecil namun konsisten secara agregat pada horizon h3–h6, tetapi belum
unggul universal. Kontribusi utamanya adalah kemampuan skenario hujan, struktur
jaringan eksplisit, dan stabilitas horizon panjang, bukan lonjakan akurasi besar.
