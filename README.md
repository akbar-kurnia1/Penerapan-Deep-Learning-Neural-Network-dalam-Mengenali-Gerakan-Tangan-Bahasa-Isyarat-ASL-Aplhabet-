# 🤟 Real-Time ASL Alphabet Recognition

Project **Deep Learning** untuk mendeteksi alfabet **American Sign Language (ASL)** secara **real-time** menggunakan webcam.
Proyek ini dibangun dengan **TensorFlow**, **OpenCV**, dan **MobileNetV2** menggunakan pendekatan **Transfer Learning**.

---

## 📋 Deskripsi Proyek

Aplikasi ini bertujuan untuk membantu komunikasi penyandang tuna rungu dengan menerjemahkan gerakan tangan alfabet **A–Z** menjadi teks secara langsung.

**Fitur utama:**

* 🔍 Deteksi alfabet ASL secara real-time
* 🧠 Model ringan & cepat dengan MobileNetV2
* 📷 Input langsung dari webcam laptop

**Spesifikasi Teknis:**

* **Model:** MobileNetV2 (Pre-trained ImageNet)
* **Dataset:** ASL Alphabet (Kaggle)
* **Akurasi:** ±95% pada data testing
* **Input:** Webcam (Real-time)

---

## 📂 Struktur Folder

```
├── realtime_asl_detection.py   # Script utama (real-time detection)
├── best_asl_model.keras        # Model hasil training
└── ASL_Training.ipynb          # (Opsional) Notebook training di Google Colab
```

---

## 🛠️ Instalasi

### 1️⃣ Clone Repository

```
git clone https://github.com/username-kamu/nama-repo.git
cd nama-repo
```

---

### 2️⃣ Buat Virtual Environment (Disarankan)

Agar dependency tidak bentrok, gunakan virtual environment.

```
python -m venv venv
```

**Aktifkan environment:**

* **Windows**

```
venv\Scripts\activate
```

* **Mac / Linux**

```
source venv/bin/activate
```

---

### 3️⃣ Install Library

Pastikan environment aktif (ada `(venv)` di terminal), lalu jalankan:

```
pip install tensorflow opencv-python numpy
```

---

## 🚀 Cara Menjalankan

1. Pastikan **webcam tidak digunakan aplikasi lain** (Zoom, Google Meet, dll).
2. Jalankan script utama:

   ```
   python realtime_asl_detection.py
   ```
3. Tunggu hingga TensorFlow selesai loading dan jendela kamera muncul.
4. Arahkan tangan ke **kotak hijau (Region of Interest)**.
5. Lihat hasil prediksi huruf dan **confidence score** di pojok kiri atas.
6. Tekan **`q`** untuk keluar dari aplikasi.

---

## ⚠️ Catatan Penting

Agar deteksi berjalan optimal, perhatikan hal berikut:

* 💡 **Pencahayaan:** Ruangan harus cukup terang
* 🧱 **Background:** Gunakan latar belakang polos
* ✋ **Posisi Tangan:** Pastikan seluruh jari masuk ke dalam kotak hijau
* 📏 **Jarak Kamera:** Jangan terlalu dekat atau terlalu jauh

---

## 👨‍💻 Author

* **Akbar Kurniawan** — *Model Training & Implementation*
* **Yudi Octavianus Siregar & Adhenn** — *Project Outline & Research*

---

## 📌 Lisensi

Proyek ini dibuat untuk memenuhi tugas project mata kuliah Kecerdasan Buatan.
Silakan dikembangkan lebih lanjut 🚀
