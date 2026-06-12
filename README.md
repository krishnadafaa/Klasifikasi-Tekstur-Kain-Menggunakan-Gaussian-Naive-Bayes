# Sistem Klasifikasi Jenis Kain Menggunakan Metode Naive Bayes Berbasis Ekstraksi Fitur Tekstur

Repositori ini berisi implementasi sistem klasifikasi jenis kain (Katun, Wol, dan Sintetis) berbasis citra digital. Aplikasi ini dikembangkan menggunakan bahasa pemrograman Python dan diimplementasikan ke dalam antarmuka web interaktif menggunakan framework Streamlit.

## Metodologi

Sistem klasifikasi ini mengintegrasikan beberapa tahapan pemrosesan citra dan pembelajaran mesin sebagai berikut:

1. **Ekstraksi Fitur Tekstur (GLCM):** Menggunakan metode *Gray Level Co-occurrence Matrix* untuk mengekstrak fitur statistik berbasis tekstur, meliputi nilai *contrast*, *homogeneity*, *energy*, *correlation*, dan *dissimilarity*.
2. **Ekstraksi Fitur Pola (LBP):** Menggunakan metode *Local Binary Pattern* untuk merepresentasikan detail pola lokal dan karakteristik mikro pada permukaan serat kain.
3. **Reduksi Dimensi (PCA):** Menggunakan *Principal Component Analysis* untuk mereduksi dimensi fitur gabungan (GLCM + LBP) dengan tetap mempertahankan variansi informasi sebesar 95% guna mengoptimalkan waktu komputasi.
4. **Klasifikasi (Gaussian Naive Bayes):** Algoritma statistik yang digunakan untuk menghitung probabilitas objek dan menentukan keputusan akhir kelas kain berdasarkan data latih yang telah dipelajari.

## Struktur Direktori

Berikut adalah struktur file utama di dalam proyek ini:

```text
Klasifikasi_Kain/
│
├── app_streamlit.py      # Script utama antarmuka web Streamlit
├── scaler.pkl            # Artefak model standardisasi data (StandardScaler)
├── pca_prep.pkl          # Artefak model reduksi dimensi (PCA)
├── model.pkl             # Artefak model klasifikasi Gaussian Naive Bayes
└── README.md             # Dokumentasi proyek
