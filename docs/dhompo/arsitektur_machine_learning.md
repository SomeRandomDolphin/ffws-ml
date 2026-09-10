# Arsitektur Model Machine Learning untuk Prediksi Tinggi Muka Air Dhompo

## 1. Gambaran Umum

Selain model deep learning, penelitian ini juga menggunakan beberapa model machine learning klasik untuk memprediksi tinggi muka air di stasiun Dhompo. Model machine learning digunakan karena mampu bekerja dengan baik pada data tabular hasil rekayasa fitur, memiliki proses training yang relatif cepat, dan lebih mudah diinterpretasikan dibanding model deep learning.

Model machine learning yang digunakan adalah:

| Model | Tipe | Keterangan Singkat |
|---|---|---|
| Linear Regression | Linear | Model baseline dengan hubungan linear antar fitur dan target |
| Ridge Regression | Linear regularized | Linear regression dengan regularisasi L2 |
| Lasso Regression | Linear regularized | Linear regression dengan regularisasi L1 dan seleksi fitur |
| Random Forest | Ensemble bagging | Gabungan banyak decision tree yang dilatih secara paralel |
| Gradient Boosting | Ensemble boosting | Gabungan decision tree yang dilatih bertahap untuk memperbaiki error |
| XGBoost | Optimized boosting | Versi gradient boosting yang lebih optimal dan memiliki regularisasi bawaan |

Tujuan utama dari model-model tersebut adalah menghasilkan prediksi tinggi muka air untuk 5 horizon waktu, yaitu 1 jam, 2 jam, 3 jam, 4 jam, dan 5 jam ke depan.

## 2. Strategi Prediksi

Pendekatan yang digunakan adalah direct multi-horizon forecasting. Artinya, setiap horizon prediksi memiliki model independen. Model untuk h1 hanya memprediksi 1 jam ke depan, model h2 hanya memprediksi 2 jam ke depan, dan seterusnya.

Skema prediksi:

```text
Input fitur pada waktu t
        |
        +--> Model h1 --> Prediksi tinggi muka air t + 1 jam
        |
        +--> Model h2 --> Prediksi tinggi muka air t + 2 jam
        |
        +--> Model h3 --> Prediksi tinggi muka air t + 3 jam
        |
        +--> Model h4 --> Prediksi tinggi muka air t + 4 jam
        |
        +--> Model h5 --> Prediksi tinggi muka air t + 5 jam
```

Pendekatan ini dipilih agar error tidak menumpuk dari satu horizon ke horizon berikutnya. Berbeda dengan pendekatan recursive, hasil prediksi h1 tidak dipakai sebagai input untuk h2. Setiap horizon langsung belajar dari fitur historis yang sama, tetapi dengan target waktu yang berbeda.

Dengan 6 algoritma dan 5 horizon, total model yang dievaluasi adalah:

```text
6 algoritma x 5 horizon = 30 model
```

## 3. Input dan Target Model

Model machine learning menerima input dalam bentuk data tabular. Setiap baris merepresentasikan satu waktu observasi, sedangkan setiap kolom merepresentasikan fitur hasil rekayasa fitur.

Bentuk input:

```text
(n_samples, n_features)
(n_samples, 160)
```

Setiap sampel memiliki 160 fitur yang dibangun dari 13 stasiun, yaitu 12 stasiun hulu dan stasiun Dhompo.

Komposisi fitur:

| Kategori Fitur | Jumlah Fitur | Penjelasan |
|---|---:|---|
| Nilai saat ini dan lag | 52 | Nilai muka air t0, t-1, t-2, dan t-3 |
| Rolling statistics | 78 | Rolling mean dan rolling standard deviation untuk 3 jam, 6 jam, dan 12 jam |
| Rate of change | 26 | Perubahan muka air antar timestep |
| Fitur temporal | 4 | Informasi jam, hari, dan indikator malam |
| Total | 160 | Total fitur input |

Target model adalah tinggi muka air Dhompo pada horizon tertentu:

| Horizon | Target |
|---|---|
| h1 | Tinggi muka air Dhompo 1 jam ke depan |
| h2 | Tinggi muka air Dhompo 2 jam ke depan |
| h3 | Tinggi muka air Dhompo 3 jam ke depan |
| h4 | Tinggi muka air Dhompo 4 jam ke depan |
| h5 | Tinggi muka air Dhompo 5 jam ke depan |

## 4. Preprocessing

Preprocessing dilakukan berbeda untuk model linear dan model tree-based.

Model linear seperti Linear Regression, Ridge, dan Lasso menggunakan StandardScaler. Scaling diperlukan karena model linear sensitif terhadap skala fitur. Tanpa scaling, fitur dengan rentang nilai besar dapat lebih dominan dibanding fitur lain.

Model tree-based seperti Random Forest, Gradient Boosting, dan XGBoost tidak menggunakan scaling. Model berbasis tree membagi data berdasarkan threshold fitur, sehingga relatif tidak sensitif terhadap perbedaan skala antar fitur.

Skema preprocessing:

```text
Fitur mentah
    |
    +--> StandardScaler --> Linear Regression / Ridge / Lasso
    |
    +--> Tanpa scaling --> Random Forest / Gradient Boosting / XGBoost
```

## 5. Arsitektur Linear Regression

Linear Regression digunakan sebagai baseline. Model ini mempelajari hubungan linear antara fitur input dan target tinggi muka air.

Rumus umum:

```text
y_pred = w1*x1 + w2*x2 + ... + wn*xn + b
```

Keterangan:

| Simbol | Penjelasan |
|---|---|
| `y_pred` | Prediksi tinggi muka air |
| `x1 ... xn` | Fitur input |
| `w1 ... wn` | Bobot model |
| `b` | Bias/intercept |

Arsitektur:

```text
Input 160 fitur
        |
StandardScaler
        |
Linear Regression
        |
Output 1 nilai prediksi untuk satu horizon
```

Karena setiap horizon memiliki model sendiri, Linear Regression dilatih sebanyak 5 kali, yaitu untuk h1 sampai h5.

## 6. Arsitektur Ridge Regression

Ridge Regression adalah pengembangan dari Linear Regression dengan regularisasi L2. Regularisasi ini menambahkan penalti terhadap bobot yang terlalu besar, sehingga model menjadi lebih stabil dan tidak terlalu mudah overfitting.

Fungsi objektif Ridge:

```text
Loss = MSE + alpha * sum(w^2)
```

Pada konfigurasi penelitian, nilai `alpha` yang digunakan adalah:

```text
alpha = 1.0
```

Arsitektur:

```text
Input 160 fitur
        |
StandardScaler
        |
Ridge Regression alpha=1.0
        |
Output 1 nilai prediksi untuk satu horizon
```

Ridge cocok digunakan ketika banyak fitur saling berkorelasi, misalnya fitur lag, rolling mean, dan nilai muka air dari beberapa stasiun yang memiliki pola mirip.

## 7. Arsitektur Lasso Regression

Lasso Regression juga merupakan model linear dengan regularisasi, tetapi menggunakan regularisasi L1. Regularisasi L1 dapat membuat sebagian bobot menjadi nol, sehingga Lasso dapat berperan sebagai seleksi fitur otomatis.

Fungsi objektif Lasso:

```text
Loss = MSE + alpha * sum(|w|)
```

Konfigurasi yang digunakan:

```text
alpha = 0.01
max_iter = 10000
```

Arsitektur:

```text
Input 160 fitur
        |
StandardScaler
        |
Lasso Regression alpha=0.01
        |
Output 1 nilai prediksi untuk satu horizon
```

Pada hasil akhir, Lasso menjadi model terbaik untuk horizon jauh h4 dan h5. Hal ini menunjukkan bahwa pada prediksi yang lebih jauh, hubungan yang lebih sederhana dan teregularisasi dapat lebih stabil dibanding model kompleks.

## 8. Arsitektur Random Forest

Random Forest adalah model ensemble yang terdiri dari banyak decision tree. Setiap tree dilatih menggunakan subset data dan subset fitur yang berbeda. Prediksi akhir diperoleh dari rata-rata prediksi seluruh tree.

Konfigurasi yang digunakan:

```text
n_estimators = 200
max_depth = 15
random_state = 42
```

Arsitektur:

```text
Input 160 fitur
        |
Decision Tree 1
Decision Tree 2
Decision Tree 3
...
Decision Tree 200
        |
Rata-rata prediksi seluruh tree
        |
Output 1 nilai prediksi untuk satu horizon
```

Random Forest cukup robust terhadap outlier dan dapat menangkap hubungan non-linear antar fitur. Model ini tidak membutuhkan scaling karena pemisahan node pada decision tree dilakukan berdasarkan threshold fitur.

## 9. Arsitektur Gradient Boosting

Gradient Boosting adalah model ensemble yang membangun decision tree secara bertahap. Tree pertama membuat prediksi awal, kemudian tree berikutnya dilatih untuk memperbaiki residual atau error dari tree sebelumnya.

Konfigurasi yang digunakan:

```text
n_estimators = 200
max_depth = 5
learning_rate = 0.1
random_state = 42
```

Arsitektur:

```text
Input 160 fitur
        |
Tree 1 menghasilkan prediksi awal
        |
Tree 2 memperbaiki error Tree 1
        |
Tree 3 memperbaiki error sebelumnya
        |
...
        |
Tree 200
        |
Penjumlahan prediksi bertahap
        |
Output 1 nilai prediksi untuk satu horizon
```

Gradient Boosting efektif untuk menangkap pola non-linear dan interaksi antar fitur. Pada hasil akhir, model ini menjadi model terbaik untuk horizon h2 dan h3.

## 10. Arsitektur XGBoost

XGBoost adalah pengembangan dari gradient boosting yang dioptimalkan dari sisi performa, regularisasi, dan efisiensi komputasi. Model ini tetap membangun tree secara bertahap, tetapi memiliki mekanisme regularisasi yang lebih kuat untuk mengurangi overfitting.

Konfigurasi yang digunakan:

```text
n_estimators = 200
max_depth = 5
learning_rate = 0.1
random_state = 42
```

Arsitektur:

```text
Input 160 fitur
        |
Boosted Tree 1
        |
Boosted Tree 2
        |
Boosted Tree 3
        |
...
        |
Boosted Tree 200
        |
Optimized additive prediction
        |
Output 1 nilai prediksi untuk satu horizon
```

XGBoost sangat baik untuk data tabular karena dapat menangkap hubungan non-linear, interaksi antar fitur, dan pola kompleks pada data. Pada hasil akhir, XGBoost menjadi model terbaik untuk horizon h1 atau prediksi 1 jam ke depan.

## 11. Model Terbaik per Horizon

Berdasarkan hasil evaluasi, model terbaik untuk setiap horizon adalah:

| Horizon | Lead Time | Model Terbaik | NSE | RMSE (m) | MAE (m) |
|---|---|---|---:|---:|---:|
| h1 | +1 jam | XGBoost | 0.9897 | 0.1242 | 0.0746 |
| h2 | +2 jam | Gradient Boosting | 0.9825 | 0.1618 | 0.0968 |
| h3 | +3 jam | Gradient Boosting | 0.9563 | 0.2562 | 0.1240 |
| h4 | +4 jam | Lasso | 0.8894 | 0.4081 | 0.1654 |
| h5 | +5 jam | Lasso | 0.7713 | 0.5877 | 0.2236 |

Pemilihan model terbaik dilakukan berdasarkan nilai NSE tertinggi pada masing-masing horizon. NSE digunakan sebagai metrik utama karena umum digunakan dalam evaluasi model hidrologi.

## 12. Arsitektur Model Produksi

Model produksi menggunakan kombinasi model terbaik dari setiap horizon. Dengan demikian, sistem tidak menggunakan satu algoritma tunggal untuk semua horizon, tetapi memilih algoritma yang paling sesuai untuk masing-masing jarak prediksi.

Skema model produksi:

```text
Input history sensor minimal 24 baris
        |
Feature engineering 160 fitur
        |
        +--> h1: XGBoost
        |
        +--> h2: Gradient Boosting
        |
        +--> h3: Gradient Boosting
        |
        +--> h4: StandardScaler + Lasso
        |
        +--> h5: StandardScaler + Lasso
        |
Output:
Prediksi h1, h2, h3, h4, h5
```

Pada horizon pendek, model tree-based lebih unggul karena mampu menangkap pola non-linear dan perubahan lokal yang kuat. Pada horizon jauh, Lasso lebih stabil karena sinyal prediktif semakin melemah dan model yang terlalu kompleks lebih berisiko overfitting.

## 13. Ringkasan

Arsitektur machine learning pada penelitian ini menggunakan pendekatan direct forecasting dengan model terpisah untuk setiap horizon prediksi. Input model berupa 160 fitur hasil rekayasa fitur dari data tinggi muka air, rolling statistics, rate of change, dan fitur temporal.

Model linear menggunakan StandardScaler, sedangkan model tree-based menggunakan fitur mentah tanpa scaling. Enam algoritma dievaluasi pada lima horizon, sehingga total terdapat 30 model kandidat.

Hasil akhir menunjukkan bahwa XGBoost paling baik untuk prediksi 1 jam, Gradient Boosting paling baik untuk prediksi 2 sampai 3 jam, dan Lasso paling stabil untuk prediksi 4 sampai 5 jam. Kombinasi ini membentuk arsitektur produksi multi-model yang disesuaikan dengan karakteristik masing-masing horizon.
