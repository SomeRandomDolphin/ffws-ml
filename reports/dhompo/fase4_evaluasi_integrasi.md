# Hasil Fase 4 — Ensemble, Evaluasi, dan Integrasi API

## Ensemble skenario hujan

Predictor hybrid Fase 3 diperluas dengan ensemble Markov-Gamma. Setiap member
menghasilkan 12 langkah hujan 30-menit, menjalankan backbone routing dari
kondisi 48 observasi terakhir, lalu melewati korektor residual. Median member
menjadi point forecast; P10/P50/P90 dilaporkan sebagai `scenario_spread`.

Implementasi mengelompokkan seluruh member menjadi satu batch inferensi per
horizon agar biaya korektor ML tidak bertambah secara linear dalam jumlah
panggilan model.

## Evaluasi probabilistik

Evaluasi memakai 110 origin test yang dipilih setiap 12 baris (6 jam) dari dua
segmen, dengan 30 skenario per origin. Routing dan korektor berasal dari train
split dengan purge gap 6 jam.

| Horizon | Coverage P10–P90 | Mean width (m) | CRPS |
|---:|---:|---:|---:|
| +1 jam | 0,0545 | 0,0070 | 0,1520 |
| +2 jam | 0,1055 | 0,0321 | 0,1751 |
| +3 jam | 0,1479 | 0,0574 | 0,1986 |
| +4 jam | 0,1958 | 0,0898 | 0,1993 |
| +5 jam | 0,2194 | 0,1557 | 0,2340 |
| +6 jam | 0,2782 | 0,2092 | 0,2374 |

Coverage nominal P10–P90 adalah 80%, tetapi coverage aktual hanya 5,5–27,8%.
Ensemble mengalami **undercoverage berat**: variasi hujan masa depan saja tidak
mencakup ketidakpastian residual, parameter routing, struktur model, dan
ketidakcocokan data generated.

Untuk Dhompo, coverage meningkat dari 0,9% pada h1 menjadi 25,5% pada h4–h6;
lebar h6 0,2258 m dan CRPS 0,2473. `Bd. Domas` paling buruk (coverage 0% pada
h1 dan 6,4% pada h6), konsisten dengan kualitas acuan yang bermasalah.

Temuan ini menyebabkan perubahan kontrak: quantile tidak disebut confidence
atau prediction interval, melainkan **scenario spread**. Kalibrasi conformal
atau residual bootstrap diperlukan sebelum interval probabilistik dapat
digunakan.

## API hybrid

Endpoint baru: `POST /predict-multistation`.

Input:

- 48 baris sejarah, interval tepat 30 menit, seluruh 15 stasiun.
- Opsional 12 nilai `future_rainfall` (mm/30 menit).
- Jika hujan tidak diberikan: `scenario_count` 2–200 dan `seed`.

Output:

- `predictions`: h1–h6 × 15 stasiun.
- `simulator_predictions`: backbone untuk transparansi.
- `scenario_spread`: P10/P50/P90 hanya pada mode ensemble.
- `future_rainfall_mode`: `provided` atau `markov_gamma_ensemble`.
- `operationally_validated=false`.
- `uncertainty_calibrated=false`.

Endpoint `/predict` lama tetap tidak berubah (Dhompo h1–h5). Startup dan
readiness kedua predictor independen; kegagalan salah satu tidak mematikan yang
lain. `/model-info` mengekspos `hybrid_multistation_ready` dan error terkait.

## Smoke test nyata

Request dengan 48 observasi terakhir 2023 berhasil:

- Ensemble 4 member: HTTP 200, enam horizon, 15 stasiun,
  `scenario_spread.h6.Dhompo` P10=9,9385, P50=9,9743, P90=10,5012 m.
- Hujan eksplisit 1 mm/30 menit selama 6 jam: HTTP 200, Dhompo h6=10,1965 m.

Nilai adalah respons skenario pada data/model generated, bukan prakiraan
operasional.

## Artefak

- Metrik: `reports/dhompo/tables/fase4_ensemble_metrics.csv`.
- Figur lokal: `reports/dhompo/figures/fase4_ensemble_coverage.png` dan
  `fase4_dhompo_interval_example.png`.
- Evaluator: `training/dhompo/evaluate_hybrid_ensemble.py`.
- Endpoint: `api/routes/multistation.py`.
- Predictor: `src/dhompo/serving/hybrid_predictor.py`.

## Reproduksi

```powershell
python training/dhompo/evaluate_hybrid_ensemble.py
uvicorn api.main:app --host 0.0.0.0 --port 8000
python -m pytest -q
```

## Kesimpulan akhir

Pipeline sekarang mampu menghasilkan prediksi/skenario 15 stasiun hingga +6
jam dan mengekspos backbone serta sebaran skenario. Point forecast hybrid
memberi peningkatan kecil pada horizon panjang, tetapi interval ensemble belum
terkalibrasi. Tanpa observasi lapangan, sistem harus tetap diposisikan sebagai
prototipe riset/surrogate dan tidak boleh dipakai untuk peringatan dini nyata.
