# Analisis Perbandingan dan Implementasi Metode Least Squares Menggunakan Eliminasi Gauss dan Dekomposisi LU

Proyek ini merupakan implementasi dan analisis perbandingan metode Eliminasi Gauss dan Dekomposisi LU untuk menyelesaikan sistem persamaan linear yang dihasilkan dari penerapan metode Least Squares pada model Lotka-Volterra.

## Deskripsi
Implementasi dilakukan dalam bahasa Python untuk menganalisis dan membandingkan kedua metode dalam hal:

- **Kompleksitas waktu**
- **Akurasi numerik**
- **Efisiensi komputasi pada multiple RHS**

## Struktur Proyek

- `implementasi.py` - Kode sumber implementasi program
- `Analisis Perbandingan dan Implementasi Metode Least Squares Menggunakan Eliminasi Gauss dan Dekomposisi LU.pdf` - Makalah lengkap yang menjelaskan teori dan hasil analisis

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Fitur
Program ini mencakup implementasi:

- Generate data fiktif model Lotka-Volterra
- Metode Eliminasi Gauss dengan partial pivoting
- Dekomposisi LU
- Perhitungan error relatif
- Visualisasi data dan hasil

## Penggunaan
Program dapat dijalankan dengan menjalankan file `implementasi.py`. Program akan:

1. Membangkitkan dataset fiktif
2. Melakukan analisis menggunakan kedua metode
3. Menampilkan visualisasi dan perbandingan hasil

## Hasil
Dari hasil pengujian diperoleh:

- Kedua metode memiliki tingkat akurasi yang sebanding dengan error relatif \(10^{-16}\) hingga \(10^{-18}\)
- Untuk kasus multiple RHS dengan \(n = 5000\), Dekomposisi LU 40% lebih cepat dibanding Eliminasi Gauss

## Author
**Muh. Rusmin Nurwadin (13523068)**  
Teknik Informatika ITB

## Referensi
Dapat dilihat pada bagian **Referensi** dalam makalah.
