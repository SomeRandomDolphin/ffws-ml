# Arsitektur Model LSTM, GRU, dan TCN untuk Prediksi Tinggi Muka Air Dhompo

## 1. Gambaran Umum

Penelitian ini menggunakan tiga arsitektur deep learning berbasis deret waktu, yaitu LSTM, GRU, dan TCN. Ketiga model digunakan untuk memprediksi tinggi muka air di stasiun Dhompo pada beberapa horizon waktu ke depan, yaitu 1 jam, 2 jam, 3 jam, 4 jam, dan 5 jam.

Setiap input model berbentuk sequence atau urutan data historis sepanjang 24 timestep. Karena interval data adalah 30 menit, maka 24 timestep merepresentasikan 12 jam data historis terakhir.

Bentuk input model adalah:

```text
(batch_size, seq_len, input_size)
(batch_size, 24, 160)
```

Keterangan:

| Komponen | Nilai | Penjelasan |
|---|---:|---|
| `batch_size` | 64 | Jumlah sampel yang diproses dalam satu iterasi training |
| `seq_len` | 24 | Panjang data historis, setara 12 jam |
| `input_size` | 160 | Jumlah fitur pada setiap timestep |
| `output_size` | 5 | Prediksi untuk horizon h1 sampai h5 |

Output model berbentuk:

```text
(batch_size, 5)
```

Artinya, setiap sampel menghasilkan lima nilai prediksi sekaligus, yaitu prediksi tinggi muka air untuk 1 sampai 5 jam ke depan.

## 2. Fitur Input

Setiap timestep terdiri dari 160 fitur yang merepresentasikan kondisi hidrologis dan temporal. Fitur tersebut dibangun dari data 13 stasiun, yaitu 12 stasiun hulu dan 1 stasiun target Dhompo.

Komposisi fitur adalah sebagai berikut:

| Kategori Fitur | Jumlah Fitur | Penjelasan |
|---|---:|---|
| Nilai saat ini dan lag | 52 | Nilai muka air pada t0, t-1, t-2, dan t-3 |
| Rolling statistics | 78 | Rolling mean dan rolling standard deviation untuk window 3 jam, 6 jam, dan 12 jam |
| Rate of change | 26 | Perubahan muka air antar timestep |
| Fitur temporal | 4 | Jam, hari, dan indikator malam hari |
| Total | 160 | Total fitur per timestep |

Dengan demikian, model tidak hanya menerima nilai muka air saat ini, tetapi juga informasi perubahan, tren historis, fluktuasi, dan pola waktu.

## 3. Arsitektur LSTM

### 3.1 Konsep LSTM

Long Short-Term Memory atau LSTM adalah pengembangan dari Recurrent Neural Network yang dirancang untuk mempelajari pola jangka panjang pada data berurutan. Pada kasus prediksi tinggi muka air, LSTM digunakan karena data hidrologi memiliki ketergantungan waktu, misalnya kenaikan air di stasiun hulu dapat memengaruhi kondisi Dhompo beberapa jam kemudian.

LSTM memiliki mekanisme gate yang mengatur informasi mana yang perlu disimpan, dilupakan, atau dikeluarkan sebagai representasi. Mekanisme ini membuat LSTM lebih stabil dalam mempelajari pola temporal dibanding RNN biasa.

### 3.2 Struktur Arsitektur

Arsitektur LSTM yang digunakan adalah:

```text
Input sequence
(batch, 24, 160)
        |
LSTM Layer 1
hidden_size = 128
        |
LSTM Layer 2
hidden_size = 128
dropout = 0.2
        |
Ambil hidden state pada timestep terakhir
(batch, 128)
        |
Fully Connected Layer
128 -> 5
        |
Output prediksi
(batch, 5)
```

Parameter utama:

| Parameter | Nilai |
|---|---:|
| Input size | 160 |
| Sequence length | 24 |
| Hidden size | 128 |
| Jumlah layer | 2 |
| Dropout | 0.2 |
| Output size | 5 |

### 3.3 Alur Kerja LSTM

Model menerima urutan data sepanjang 12 jam terakhir. Setiap timestep berisi 160 fitur. Data tersebut diproses oleh dua layer LSTM bertumpuk. Layer pertama mempelajari pola temporal dasar, sedangkan layer kedua membentuk representasi temporal yang lebih abstrak.

Setelah seluruh sequence diproses, model mengambil hidden state pada timestep terakhir. Hidden state ini dianggap sebagai ringkasan informasi dari 12 jam data historis. Ringkasan tersebut kemudian dikirim ke fully connected layer untuk menghasilkan lima prediksi horizon.

## 4. Arsitektur GRU

### 4.1 Konsep GRU

Gated Recurrent Unit atau GRU adalah arsitektur recurrent neural network yang mirip dengan LSTM, tetapi memiliki struktur gate yang lebih sederhana. GRU tetap mampu mempelajari pola temporal, tetapi dengan jumlah parameter yang lebih sedikit dibanding LSTM.

Pada penelitian ini, GRU digunakan sebagai pembanding LSTM untuk melihat apakah model recurrent yang lebih sederhana tetap mampu menghasilkan prediksi yang baik pada data tinggi muka air.

### 4.2 Struktur Arsitektur

Arsitektur GRU yang digunakan adalah:

```text
Input sequence
(batch, 24, 160)
        |
GRU Layer 1
hidden_size = 128
        |
GRU Layer 2
hidden_size = 128
dropout = 0.2
        |
Ambil hidden state pada timestep terakhir
(batch, 128)
        |
Fully Connected Layer
128 -> 5
        |
Output prediksi
(batch, 5)
```

Parameter utama:

| Parameter | Nilai |
|---|---:|
| Input size | 160 |
| Sequence length | 24 |
| Hidden size | 128 |
| Jumlah layer | 2 |
| Dropout | 0.2 |
| Output size | 5 |

### 4.3 Alur Kerja GRU

GRU memproses input secara berurutan dari timestep pertama sampai timestep ke-24. Setiap timestep memperbarui hidden state berdasarkan informasi saat ini dan informasi sebelumnya.

Berbeda dengan LSTM yang memiliki cell state dan hidden state, GRU hanya menggunakan hidden state. Hal ini membuat GRU lebih ringan secara komputasi. Setelah sequence selesai diproses, hidden state terakhir digunakan sebagai representasi kondisi historis, lalu dimasukkan ke fully connected layer untuk menghasilkan prediksi h1 sampai h5.

## 5. Arsitektur TCN

### 5.1 Konsep TCN

Temporal Convolutional Network atau TCN adalah model deep learning untuk data deret waktu yang menggunakan operasi konvolusi satu dimensi. Berbeda dengan LSTM dan GRU yang membaca data secara recurrent, TCN mempelajari pola temporal menggunakan filter konvolusi sepanjang dimensi waktu.

TCN cocok digunakan untuk prediksi time series karena dapat menangkap pola lokal dan tren temporal tanpa harus memproses data secara berurutan seperti RNN. Hal ini membuat TCN lebih mudah diparalelkan.

### 5.2 Struktur Arsitektur

Arsitektur TCN yang digunakan terdiri dari beberapa blok konvolusi temporal:

```text
Input sequence
(batch, 24, 160)
        |
Transpose untuk Conv1D
(batch, 160, 24)
        |
TCN Block 1
Conv1D: 160 -> 64, kernel_size = 3
        |
TCN Block 2
Conv1D: 64 -> 64, kernel_size = 3
        |
TCN Block 3
Conv1D: 64 -> 64, kernel_size = 3
        |
TCN Block 4
Conv1D: 64 -> 64, kernel_size = 3
        |
Ambil representasi temporal akhir
(batch, 64)
        |
Fully Connected Layer
64 -> 5
        |
Output prediksi
(batch, 5)
```

Parameter utama:

| Parameter | Nilai |
|---|---:|
| Input size | 160 |
| Sequence length | 24 |
| Jumlah blok TCN | 4 |
| Channel hidden | 64 |
| Kernel size | 3 |
| Output size | 5 |

### 5.3 Alur Kerja TCN

Pada TCN, fitur input diperlakukan sebagai channel, sedangkan urutan waktu diproses melalui Conv1D. Kernel konvolusi bergerak sepanjang dimensi waktu untuk menangkap pola perubahan muka air dari beberapa timestep berdekatan.

Setiap blok TCN menghasilkan representasi temporal yang semakin abstrak. Setelah melewati empat blok konvolusi, representasi akhir digunakan oleh fully connected layer untuk menghasilkan lima prediksi horizon.

## 6. Perbandingan Arsitektur

| Aspek | LSTM | GRU | TCN |
|---|---|---|---|
| Jenis model | Recurrent | Recurrent | Convolutional |
| Cara membaca waktu | Berurutan | Berurutan | Konvolusi sepanjang waktu |
| Kompleksitas | Lebih tinggi | Lebih ringan dari LSTM | Relatif efisien |
| Memori temporal | Cell state dan hidden state | Hidden state | Receptive field konvolusi |
| Output | Multi-horizon langsung | Multi-horizon langsung | Multi-horizon langsung |
| Cocok untuk | Pola temporal kompleks | Pola temporal dengan model lebih sederhana | Pola lokal dan tren temporal |

Secara umum, LSTM dan GRU mempelajari dependensi waktu melalui mekanisme recurrent, sedangkan TCN mempelajari pola waktu melalui filter konvolusi. LSTM cenderung lebih lengkap karena memiliki cell state, GRU lebih sederhana dan efisien, sedangkan TCN memiliki keunggulan dalam pemrosesan paralel.

## 7. Ringkasan

Ketiga model menggunakan input dan output yang sama, yaitu sequence sepanjang 24 timestep dengan 160 fitur per timestep, dan menghasilkan lima prediksi tinggi muka air untuk horizon 1 sampai 5 jam ke depan.

LSTM dan GRU menggunakan pendekatan recurrent dengan hidden size 128 dan dua layer. Perbedaannya terletak pada mekanisme internal: LSTM menggunakan cell state dan beberapa gate, sedangkan GRU menggunakan struktur gate yang lebih sederhana.

TCN menggunakan pendekatan konvolusi temporal dengan empat blok Conv1D. Model ini memproses pola waktu melalui filter konvolusi, bukan melalui pembacaan recurrent.

Dengan membandingkan ketiga arsitektur ini, penelitian dapat mengevaluasi pendekatan mana yang paling sesuai untuk karakteristik data tinggi muka air Dhompo.
