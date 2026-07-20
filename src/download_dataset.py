"""
download_dataset.py
Mengunduh dan menyusun dataset "Indonesian Vehicle License Plate Dataset" dari Kaggle.

Syarat:
1. Buat akun Kaggle -> Settings -> Create New API Token -> kaggle.json terunduh.
2. Letakkan file kaggle.json di:
   - Linux/Mac : ~/.kaggle/kaggle.json
   - Windows   : C:\\Users\\<user>\\.kaggle\\kaggle.json
3. pip install kaggle (sudah ada di requirements.txt)
"""

import os
import zipfile
from pathlib import Path

DATASET_SLUG = "linkgish/indonesian-plate-number-from-multi-sources"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as e:
        raise SystemExit(
            "Kaggle API token tidak ditemukan. Pastikan kaggle.json sudah "
            "diletakkan di ~/.kaggle/kaggle.json (lihat docstring di atas)."
        ) from e

    api = KaggleApi()
    api.authenticate()

    print(f"Mengunduh dataset: {DATASET_SLUG} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(RAW_DIR), quiet=False)

    # Ekstrak semua file zip yang terunduh
    for zip_path in RAW_DIR.glob("*.zip"):
        print(f"Mengekstrak {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()  # hapus zip setelah diekstrak

    print(f"Selesai. Dataset tersedia di: {RAW_DIR}")
    print("Isi folder:")
    for item in sorted(RAW_DIR.iterdir()):
        print(" -", item.name)


if __name__ == "__main__":
    download_dataset()
