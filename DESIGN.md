# Arah desain ITS Water Dashboard

## Status dan tujuan

Dokumen ini menetapkan target penataan frontend yang disetujui pengguna: lebih terang dan netral, dengan kepadatan informasi seimbang. Nilai di bawah adalah spesifikasi untuk implementasi berikutnya, bukan laporan bahwa UI saat ini sudah memenuhinya.

Cakupan meliputi dashboard peta, panel pendukung, grafik, dan detail stasiun Welang serta Surabaya. Pengguna acuan adalah peneliti dan pengguna pemantauan yang mencari stasiun, membaca kondisi muka air, dan memeriksa riwayat. Pedoman ini tidak menyatakan aplikasi sebagai sistem peringatan operasional yang telah tervalidasi.

Pertahankan identitas ITS Water Dashboard, logo yang tersedia, dan struktur navigasi. Tema terang menjadi target tahap ini sesuai pilihan pengguna. Penambahan tema gelap memerlukan cakupan pekerjaan tersendiri; pilihan basemap gelap tidak berarti tema antarmuka berubah.

## Karakter dan hierarki

Karakter visual: terang, tenang, teknis, dan mudah dipindai. Peta beserta penanda stasiun menjadi ciri utama aplikasi; panel membantu membaca lokasi serta data yang dipilih.

- Dashboard mengutamakan peta. Pencarian, pilihan wilayah, dan kontrol lapisan tetap mudah dijangkau.
- Detail stasiun mengutamakan nama, pembacaan utama, status, satuan, dan waktu pembaruan; grafik dan data pendukung mengikuti.
- Gunakan struktur berulang untuk fungsi yang sama. Peringatan tetap lebih menonjol daripada keterangan waktu.
- Header dan navigasi menggunakan permukaan putih dengan teks gelap. Gunakan navy pada identitas dan teks secara terbatas agar header tidak bersaing dengan peta.
- ENERGY 1: penekanan melalui hierarki teks dan status, tanpa dekorasi yang bersaing dengan data.
- RHYTHM 1: susunan dan jarak teratur untuk membantu pemindaian berulang.
- MOTION 1: transisi singkat pada kontrol serta buka/tutup panel; tanpa animasi dekoratif berulang.

## Warna

Nama token berikut adalah target; implementasi dapat memetakan variabel lama ke token ini selama migrasi.

| Token | Nilai | Penggunaan dan alasan |
|---|---|---|
| `--color-bg` | `#F3F6F8` | Latar aplikasi yang membedakan area kerja dari panel |
| `--color-surface` | `#FFFFFF` | Header, panel, input, dan tooltip |
| `--color-surface-hover` | `#EDF2F5` | Hover netral serta latar bagian pendukung |
| `--color-text` | `#172B3A` | Teks utama dengan kontras kuat pada permukaan terang |
| `--color-text-muted` | `#52636E` | Keterangan dan satuan yang tetap terbaca |
| `--color-divider` | `#DCE4E9` | Pemisah dekoratif antarbagian |
| `--color-control-border` | `#7B8D99` | Batas input dan kontrol yang perlu dikenali |
| `--color-action` | `#246B91` | Aksi utama, tautan, fokus, dan pilihan aktif |
| `--color-action-hover` | `#174F6E` | Hover aksi utama |
| `--color-selection-bg` | `#EAF2F7` | Latar pilihan aktif dengan label/batas biru |

Tombol utama memakai teks putih di atas warna aksi. Kontrol sekunder memakai permukaan putih dan teks gelap atau biru. Pemisah dekoratif tidak boleh menjadi satu-satunya penanda batas kontrol. Jangan menggunakan aqua/mint sebagai aksen tambahan tanpa fungsi data yang jelas.

### Status muka air

Pertahankan empat kategori yang sudah ada. Tabel ini mencatat warna marker dari `frontend/src/lib/demo-data.ts` dan pasangan teks/latar dari `frontend/src/app/globals.css` sebagai dasar penyatuan. Warna marker tidak otomatis cocok sebagai warna teks; uji penggunaannya pada permukaan sebenarnya.

| Status | Marker | Teks | Latar badge |
|---|---|---|---|
| Normal | `#32956B` | `#247855` | `#E5F3EC` |
| Meningkat | `#D9AE32` | `#806714` | `#FBF3D8` |
| Waspada | `#E48732` | `#A95812` | `#FCF0E2` |
| Bahaya | `#CB4949` | `#B03B3B` | `#FBEAEA` |

Gunakan pemetaan yang sama pada marker, legenda, tabel, dan badge; bedakan peran warna marker dengan teks. Pertahankan label dan gunakan garis tepi/halo marker agar dapat dibaca di atas basemap berbeda. Perubahan visual tidak boleh mengubah ambang atau kategori data.

Kualitas air memakai kategori Baik, Sedang, Buruk yang terpisah dari status muka air. Warna dasar yang tersedia adalah `#28998D`, `#718093`, `#765477`; gunakan label serta bentuk/garis pembeda pada legenda. Validasi kontras sebelum menggunakannya untuk teks.

Kesegaran/koneksi data memakai label seperti Live, Data kedaluwarsa, atau Koneksi terputus sesuai keadaan sumber. Warna hijau koneksi tidak menyatakan muka air Normal. Pertahankan keadaan lain yang disediakan sumber, termasuk waktu pembacaan di masa depan. Warna semantik data dan lapisan geografis memiliki fungsi tersendiri di luar aksen antarmuka.

## Tipografi

Pertahankan IBM Plex Sans untuk label, isi, judul, dan kontrol karena sudah menjadi fondasi antarmuka. IBM Plex Mono digunakan terbatas untuk pembacaan utama atau koordinat agar angka mudah diperiksa. Gunakan fallback `sans-serif` dan `monospace` bila font tidak tersedia.

| Peran | Ukuran | Bobot |
|---|---|---|
| Keterangan dan waktu | 12px | 400 |
| Label dan tabel | 13px | 500 untuk label/header, 400 untuk isi |
| Isi dan kontrol | 14px | 400 untuk isi, 500 untuk kontrol |
| Subjudul | 16px | 600 |
| Judul panel | 20px | 600 |
| Judul halaman | 28px | 600 |
| Pembacaan utama | 32px | 500 |

Tinggi baris isi 1,5 dan judul 1,25. Gunakan `font-variant-numeric: tabular-nums` untuk pembacaan, tabel angka, waktu, dan perubahan nilai. Satuan menggunakan ukuran label atau isi. Hindari teks informasi di bawah 12px dan uppercase dengan tracking lebar. Nama stasiun panjang dapat membungkus tanpa menutupi kontrol.

## Ukuran, permukaan, dan komponen

Skala spasi: 4, 8, 12, 16, 24, 32px. Gunakan 8px di dalam kelompok kontrol, 16px antarbagian terkait, dan 24/32px untuk pemisahan bagian utama. Padding panel 20px desktop dan 16px mobile merupakan pengecualian tetap untuk kepadatan seimbang.

Radius badge 4px, kontrol 8px, panel 12px. Marker geografis tetap bundar. Panel biasa menggunakan pemisah atau border; bayangan hanya untuk popup, dropdown, dan panel yang menumpuk di atas konten. Target bayangan overlay: `0 8px 24px rgb(23 43 58 / 12%)`.

| Komponen | Aturan bersama |
|---|---|
| Tombol utama | Tinggi minimal 40px desktop, padding horizontal 16px, warna aksi; hover memakai warna aksi hover |
| Tombol sekunder | Ukuran sama; putih, border kontrol, hover netral |
| Tombol ikon | Area 44 x 44px, ikon 18/20px sesuai konteks, nama aksesibel yang menjelaskan aksi |
| Input/select | Minimal 40px desktop, font 14px, padding horizontal 12px, label jelas, border kontrol |
| Tab/pilihan wilayah | Ukuran kontrol konsisten; pilihan aktif memakai latar pilihan, label biru, dan penanda tambahan seperti border |
| Panel | Header konsisten, judul 20px, tombol tutup 44px, isi mengikuti padding panel |
| Badge | Label 12px/600, padding 4px 8px, radius 4px; selalu menyebut status |
| Tooltip | Putih, teks 12px, padding 12px, radius 8px; tampilkan nama, nilai, satuan, dan waktu yang relevan |
| Tabel | Header/isi 13px, padding sel 12px, angka rata kanan, pemisah baris tipis, hover netral |

Semua kontrol memiliki keadaan default, hover, aktif/terpilih bila relevan, fokus, dan disabled. Gunakan outline fokus biru solid 3px dengan offset 3px; jangan menghapusnya tanpa pengganti. Disabled memakai permukaan netral, teks sekunder, dan semantik disabled tanpa respons klik. Sediakan alasan ketika ketidaktersediaan aksi perlu dijelaskan.

Gunakan ikon yang tersedia dengan arti konsisten. Perubahan warna status, arah tren, atau ukuran ikon harus memiliki fungsi informasi. Hindari menambahkan dekorasi untuk mengisi ruang kosong.

## Peta, grafik, dan data

- Pertahankan basemap serta lapisan yang tersedia. Konsistensi antarmuka tidak mengharuskan semua lapisan geografis menjadi satu warna.
- Selaraskan warna seri dan legenda antargrafik untuk besaran yang sama. Bedakan observasi, prediksi, serta ambang melalui label dan pola garis, bukan warna saja.
- Grafik memiliki judul yang menyebut besaran/rentang waktu dan sumbu dengan satuan. Tooltip dan tabel menggunakan format nilai yang sama.
- Tampilkan satuan sesuai sumber. Welang menggunakan meter; bila Surabaya masih memiliki asumsi cm, tampilkan asumsi tersebut sampai terverifikasi. Konversi satuan bukan bagian penataan visual.
- Penanda simulasi tetap terlihat pada area data terkait. Label live memerlukan keadaan sumber yang mendukungnya; animasi atau titik hijau tidak cukup.
- Waktu pembaruan mengikuti data dan zona waktu yang jelas. Data lama tidak diberi timestamp baru seolah pembacaan baru.
- Loading menyebut data yang sedang dimuat. Keadaan kosong membedakan belum ada data dengan hasil pencarian kosong. Error menyebut kegagalan dan menyediakan tindakan pemulihan hanya bila tindakan tersebut tersedia.
- Jika pembacaan terakhir tetap ditampilkan ketika koneksi gagal, tandai sebagai data terakhir beserta waktunya. Jangan mengganti pembacaan hilang dengan angka nol.

## Responsivitas dan aksesibilitas

Pada lebar maksimal 700px, gunakan padding mobile dan area sentuh minimal 44px untuk kontrol. Panel overlay harus tetap berada di viewport dan isi dapat digulir tanpa menyembunyikan tombol tutup. Tabel lebar boleh menggulir di dalam pembungkusnya; halaman tidak boleh overflow horizontal. Pada ukuran lebih besar, susunan mengikuti ruang tersedia dengan hierarki yang sama.

Semua kontrol dapat digunakan dengan keyboard dan memiliki fokus terlihat. Panel yang mendukung Escape mengembalikan fokus ke pemicunya. Dialog modal menjaga fokus di dalamnya; panel nonmodal tidak membuat perangkap fokus. Informasi tooltip penting juga tersedia melalui fokus atau teks panel.

Target kontras seluruh teks minimal 4,5:1. Batas kontrol dan informasi grafis penting ditargetkan minimal 3:1 terhadap warna yang bersebelahan. Periksa kombinasi aktual, termasuk badge dan marker di atas basemap; daftar hex bukan bukti lulus kontras.

Gunakan transisi 160ms untuk warna kontrol dan maksimal 220ms untuk buka/tutup panel. Hormati `prefers-reduced-motion` dengan mematikan animasi/transisi yang tidak diperlukan. Pergerakan kamera peta akibat aksi pengguna tetap fungsional dan mengikuti preferensi reduced motion.

## Urutan penerapan dan penerimaan

Terapkan bertahap: token warna dan tipografi, komponen berulang, jarak/kepadatan panel, lalu legenda/tooltip/grafik/peta. Satukan fondasi visual dashboard dan kedua jenis halaman detail tanpa mengubah logika datanya.

Untuk implementasi UI berikutnya:

- Jalankan build frontend dan catat hasilnya.
- Periksa lebar 375, 768, dan 1440px; teks panjang, panel terbuka, tabel, dan kontrol harus tetap dapat digunakan.
- Periksa pencarian, wilayah, lapisan, legenda, popup, grafik, detail stasiun, dan kembali ke peta.
- Periksa loading, kosong, error, data kedaluwarsa, serta perbedaan simulasi/live yang relevan.
- Periksa keyboard, fokus, Escape, reduced motion, kontras, dan error console.
- Catat hasil aktual dan keterbatasan. Jangan menandai pemeriksaan yang belum dilakukan sebagai lulus.

Pembuatan pedoman ini hanya diverifikasi sebagai perubahan dokumentasi. Tidak ada klaim bahwa pemeriksaan aplikasi di atas telah dijalankan untuk target desain ini.
