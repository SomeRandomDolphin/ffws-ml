# Dataset

| Lokasi lokal | Isi |
|---|---|
| `data/` | Dataset Dhompo: `data-raw.xlsx`, `data-clean.csv`, dan `Data generated 2023.xlsx` |
| `data/surabaya/` | Dataset Surabaya: `ketinggian_30menit_wide.csv` |

Data Dhompo menggunakan meter, Surabaya sentimeter. Keduanya memakai observasi 30 menit dengan schema berbeda. Data generated 2023 merupakan sumber eksperimen tersendiri dan jangan diperlakukan sebagai pengganti data mentah.

Konfigurasi sumber data ada di `configs/dhompo/training.yaml` dan `configs/surabaya/urban_water_level*.yaml`. Dataset baru diabaikan Git agar tidak ikut dipush; berkas Dhompo yang sudah lama dilacak tetap berada di lokasi lamanya. Berkas sumber tidak boleh ditimpa oleh preprocessing.

Baca [panduan Dhompo](../docs/dhompo/index.md) atau [panduan Surabaya](../docs/surabaya/index.md) sebelum menjalankan eksperimen.
