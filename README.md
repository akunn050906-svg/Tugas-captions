# OCR Plat Nomor Kendaraan Indonesia

Capstone Project — Mata Kuliah Kecerdasan Buatan (Computer Vision)
Universitas Bale Bandung (UNIBBA)

**Nama:** Najib Fadhil Akbar
**NIM:** 301240019
**Kelas:** Teknik Informatika 4A

## 1. Deskripsi Proyek

Sistem end-to-end untuk **mendeteksi dan membaca teks plat nomor kendaraan Indonesia**
dari sebuah gambar. Pipeline terdiri dari dua tahap:

1. **Deteksi** — menemukan lokasi/bounding box plat nomor pada gambar kendaraan
   menggunakan **YOLOv8**.
2. **Pengenalan Teks (OCR)** — membaca karakter pada plat yang sudah di-crop,
   dengan membandingkan dua pendekatan: **Tesseract OCR** vs **EasyOCR**.

Perbandingan dua pendekatan OCR ini digunakan untuk memenuhi ketentuan capstone
(minimal 2 pendekatan dibandingkan) sekaligus untuk memilih engine terbaik yang
dipakai pada aplikasi web demo (Flask).

## 2. Dataset

**Indonesian Vehicle License Plate Dataset** (Kaggle)
Sumber: https://www.kaggle.com/datasets/linkgish/indonesian-plate-number-from-multi-sources

Dataset terdiri dari dua bagian:
- `plate_detection_dataset/` — 1.383 gambar kendaraan + anotasi COCO (bounding box plat), untuk training YOLOv8.
- `plate_text_dataset/` — 1.863 gambar crop plat nomor + label teks (`label.csv`), untuk evaluasi OCR.

## 3. Struktur Folder

```
ocr-plat-nomor/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # dataset asli hasil download (tidak di-commit, lihat .gitignore)
│   └── processed/            # hasil crop plat, label bersih, split train/val
├── src/
│   ├── download_dataset.py   # download & susun dataset dari Kaggle
│   ├── preprocess.py         # konversi anotasi COCO -> format YOLO, split data
│   ├── train_detection.py    # training YOLOv8 untuk deteksi plat
│   ├── evaluate_detection.py # evaluasi mAP, precision, recall model deteksi
│   ├── ocr_compare.py        # bandingkan Tesseract vs EasyOCR (akurasi, CER)
│   └── pipeline.py           # pipeline end-to-end: deteksi -> crop -> OCR
├── models/                   # bobot model hasil training (.pt) - tidak di-commit
├── notebooks/
│   └── eksplorasi.ipynb      # EDA dataset (opsional)
└── app/
    ├── app.py                # aplikasi web Flask untuk demo
    ├── templates/
    │   └── index.html
    └── static/uploads/        # tempat sementara gambar yang diupload user
```

## 4. Cara Menjalankan

### 4.1 Setup environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Catatan: gunakan Python 3.11 (bukan 3.13/3.14) agar kompatibel dengan
> TensorFlow/PyTorch, sesuai pengalaman project-project sebelumnya.

### 4.2 Download dataset
```bash
# siapkan kaggle.json (API token) di ~/.kaggle/kaggle.json terlebih dahulu
python src/download_dataset.py
```

### 4.3 Preprocessing (COCO -> format YOLO, split train/val/test)
```bash
python src/preprocess.py
```

### 4.4 Training model deteksi (YOLOv8)
```bash
python src/train_detection.py --epochs 50 --imgsz 640
```

### 4.5 Evaluasi model deteksi
```bash
python src/evaluate_detection.py --weights models/best.pt
```

### 4.6 Perbandingan Tesseract vs EasyOCR
```bash
python src/ocr_compare.py
```

### 4.7 Coba pipeline end-to-end lewat CLI
```bash
python src/pipeline.py --image path/ke/gambar.jpg
```

### 4.8 Jalankan aplikasi web (Flask)
```bash
cd app
python app.py
```
Buka `http://localhost:5000` di browser.

## 5. Metodologi Singkat

| Tahap | Model/Metode | Metrik Evaluasi |
|---|---|---|
| Deteksi Plat | YOLOv8n (perbandingan vs YOLOv8s) | mAP@0.5, Precision, Recall |
| OCR / Recognition | Tesseract OCR vs EasyOCR | Character Error Rate (CER), Accuracy exact-match |

## 6. Rencana Perbandingan (sesuai ketentuan capstone: minimal 2 pendekatan)
1. **Deteksi**: YOLOv8n (ringan, cepat) vs YOLOv8s (lebih besar, lebih akurat)
2. **OCR**: Tesseract (rule-based, ringan) vs EasyOCR (deep learning, CRNN-based)

## 7. Deployment
Aplikasi Flask direncanakan di-deploy ke **Railway**, mengikuti pola deployment
project-project sebelumnya (CurrencyVision, CloudVision, TBO Capstone).

## 8. Lisensi & Sumber
- Dataset: Kaggle — linkgish/indonesian-plate-number-from-multi-sources
- Model deteksi: Ultralytics YOLOv8 (AGPL-3.0)
- OCR: Tesseract OCR (Apache 2.0), EasyOCR (Apache 2.0)
