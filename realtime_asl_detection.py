import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = 'best_asl_model.keras'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5 

LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'del', 'nothing', 'space'
]

def load_trained_model():
    """Memuat model Keras yang sudah dilatih."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: File model '{MODEL_PATH}' tidak ditemukan!")
        print("   Pastikan file .keras berada di folder yang sama dengan script ini.")
        return None
    
    print("🔄 Sedang memuat model... (ini mungkin memakan waktu beberapa detik)")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model berhasil dimuat.")
        return model
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        return None

def main():
    model = load_trained_model()
    if model is None:
        return

    # Inisialisasi Webcam (Index 0 biasanya webcam default laptop)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Tidak dapat membuka webcam.")
        return

    print("🎥 Webcam aktif. Tekan 'q' untuk keluar.")
    print("ℹ️  Letakkan tangan Anda di dalam kotak hijau.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame dari webcam.")
            break

        # Flip frame secara horizontal agar seperti cermin (opsional, tapi lebih natural)
        frame = cv2.flip(frame, 1)

        # Dapatkan dimensi frame
        h, w, _ = frame.shape

        # --- 1. Definisikan Region of Interest (ROI) ---
        # Kita buat kotak di kanan atas layar untuk area deteksi tangan
        # Koordinat: (x1, y1) top-left, (x2, y2) bottom-right
        roi_size = 300
        x1, y1 = int(w * 0.5), 50  # Posisi agak ke tengah-kanan
        x2, y2 = x1 + roi_size, y1 + roi_size

        # Pastikan ROI tidak keluar batas frame
        if x2 > w: x2 = w
        if y2 > h: y2 = h

        # Gambar kotak ROI di frame asli (Warna Hijau)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Area Tangan", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- 2. Preprocessing ROI ---
        # Potong gambar sesuai area ROI
        roi = frame[y1:y2, x1:x2]

        if roi.size != 0: # Cek jika ROI valid
            # Resize ke 224x224 sesuai input MobileNetV2
            img_input = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            
            # Ubah warna BGR (OpenCV) ke RGB (TensorFlow/Keras standard)
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            
            # Normalisasi piksel (0-255 -> 0-1)
            img_input = img_input.astype('float32') / 255.0
            
            # Tambahkan dimensi batch: (224, 224, 3) -> (1, 224, 224, 3)
            img_input = np.expand_dims(img_input, axis=0)

            # --- 3. Prediksi ---
            preds = model.predict(img_input, verbose=0)
            idx = np.argmax(preds) # Ambil index dengan probabilitas tertinggi
            confidence = preds[0][idx]
            
            label_text = LABELS[idx]

            # --- 4. Tampilkan Hasil ---
            # Tampilkan label hanya jika confidence di atas threshold
            if confidence > CONFIDENCE_THRESHOLD:
                display_text = f"{label_text} ({confidence*100:.1f}%)"
                color = (0, 255, 0) # Hijau jika yakin
            else:
                display_text = "Tidak Yakin"
                color = (0, 0, 255) # Merah jika tidak yakin

            # Tampilkan teks hasil prediksi di atas kotak ROI
            cv2.putText(frame, f"Prediksi: {display_text}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Tampilkan frame utama
        cv2.imshow('ASL Detection Real-time', frame)

        # Tombol keluar ('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Program selesai.")

if __name__ == "__main__":
    main()