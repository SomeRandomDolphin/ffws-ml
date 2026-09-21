# Hasil Fase 2 — Simulator Sintetis 15 Stasiun

## Ruang lingkup

Fase 2 membangun simulator skenario untuk seluruh 15 stasiun pada
`configs/dhompo/network.yaml`. Simulator tidak diklaim sebagai model hidraulik
penuh karena data yang tersedia adalah muka air `SP (meters)`, bukan debit atau
rating curve, dan seluruh data acuan berasal dari software.

Komponen simulator:

1. Generator hujan **Markov-Gamma** pada interval 30 menit.
2. Reservoir hujan lokal per stasiun.
3. Transfer anomali level hulu-hilir dengan lag dan atenuasi per reach.
4. Noise proses kecil berdasarkan residual dinamika setiap stasiun.

## Kalibrasi hujan

Kalibrasi dari `Data generated 2023.xlsx` menghasilkan:

| Parameter | Nilai |
|---|---:|
| P(basah \| kering) | 0,0284 |
| P(basah \| basah) | 0,6183 |
| Gamma shape | 0,3631 |
| Gamma scale | 9,2722 |

Shape Gamma di bawah satu menghasilkan banyak kejadian kecil dengan ekor hujan
lebat, konsisten dengan data acuan. Multiplikator bulanan adalah prior eksplisit
karena data hanya mencakup Januari–Maret dan tidak cukup untuk mengestimasi 12
bulan secara empiris.

## Formulasi routing

Level dinyatakan sebagai baseline median + anomali. Untuk stasiun `i`:

```text
a_i(t) = recession_i * a_i(t-1)
       + (1 - recession_i) * upstream_i(t-lag)
       + rainfall_gain_i * rain_state(t)
       + process_noise_i(t)
```

`recession` dan noise dikalibrasi dari dinamika level kedua segmen. Atenuasi
reach diestimasi dari regresi anomali hulu tertunda terhadap anomali hilir dan
dibatasi 0,05–0,95 untuk menjaga stabilitas. Respons hujan memakai prior skala
berdasarkan standar deviasi stasiun karena hujan acuan hampir tidak berkorelasi
dengan level.

## Dataset 10 tahun

- Lokasi lokal: `data/synthetic/hydro_synthetic.csv`.
- Rentang: 2024-01-01 sampai 2033-12-31, interval 30 menit.
- Ukuran: 175.320 baris, 15 stasiun + hujan + `scenario_id`.
- Wet fraction hujan: 6,96% (acuan 2023: 6,87%).
- Hujan maksimum: 50 mm/30 menit (cap konfigurasi).
- Semua level finite dan non-negatif.
- Dhompo: mean 9,872 m, std 0,606 m, q99 12,311 m, maksimum 16,500 m.
- Korelasi hujan–Dhompo meningkat dari 0,113 pada lag 0 menjadi sekitar 0,280
  pada lag 3 jam, lalu menurun; ini menghasilkan respons tertunda yang tidak ada
  pada data generated 2023.

Variabilitas sintetis terhadap acuan tidak seragam. Rasio standar deviasi
Dhompo adalah 0,646; beberapa stasiun halus berada sekitar 0,9–1,6, sedangkan
stasiun acuan noisy (`AWLR Kademungan`, `Bd. Domas`) hanya sekitar 0,12–0,13.
Ini disengaja: simulator tidak meniru white noise acuan secara penuh.

## Artefak dan reproduksi

- Konfigurasi: `configs/dhompo/simulator.yaml`.
- Parameter hasil kalibrasi: `configs/dhompo/simulator_calibrated.yaml`.
- Ringkasan: `reports/dhompo/tables/fase2_synthetic_summary.csv`.
- Generator: `src/dhompo/data/rainfall_sim.py`.
- Routing: `src/dhompo/data/routing_sim.py`.
- API skenario: `src/dhompo/data/scenarios.py`.

```powershell
python scripts/build_synthetic.py --calibrate-only
python scripts/build_synthetic.py --periods 1440 --output data/synthetic/smoke.csv
python scripts/build_synthetic.py
python -m pytest -q
```

## Batasan

- Tidak ada validasi terhadap observasi lapangan.
- Routing dilakukan pada anomali muka air, bukan debit; mass balance hidrologi
  tidak dapat diklaim.
- Topologi dan waktu tempuh adalah asumsi eksplisit dari konfigurasi proyek.
- Musiman hujan di luar Januari–Maret memakai prior, bukan estimasi data.

## Keputusan ke Fase 3

Dataset ini tersedia untuk stress-test, analisis skenario, dan augmentasi pada
eksperimen lanjutan. Fase 3 memakai simulator sebagai backbone langsung, tetapi
tidak menjadikan output sintetis simulator sebagai target residual: residualnya
secara konstruksi mendekati nol dan akan memberi evaluasi semu. Korektor belajar
dari data acuan train; evaluasi tetap dibandingkan dengan persistence dan
HistGradientBoosting Fase 1.
