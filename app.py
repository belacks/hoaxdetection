import streamlit as st
import pandas as pd
import numpy as np
import joblib # Untuk load model sklearn/meta
import torch # Diperlukan oleh transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os
import time
import gc
import gdown # Untuk download dari Google Drive
import zipfile # Untuk unzip folder IndoBERT
import shutil # Untuk menghapus file zip setelah ekstrak

# --- Konfigurasi Path & Google Drive ID ---

# Ganti dengan ID File Google Drive Anda!
GDRIVE_ID_INDOBERT_ZIP = "1xPDhZoEamuIH6-r3kcjAC_FKVXh9UDpe" # ID file final_indobert_model.zip
GDRIVE_ID_LOGREG_TFIDF = "1KjvwuZ5IbRbeSbxXhSF57qgMs4_G8R3Z" # ID file tfidf_logreg_pipeline_final.joblib
GDRIVE_ID_MNB_TFIDF = "1KeMmV1zqzxzyihsVBKrQRoLR_gmUgc0S" # ID file tfidf_mnb_pipeline_final.joblib
GDRIVE_ID_META_MODEL = "1KcnnwygROrnYcbPwE6KcOnztxGmUppwf" # ID file meta_model_final.joblib
# GDRIVE_ID_SCALER = "YOUR_SCALER_FILE_ID" # Jika pakai scaler

# Path lokal di environment Streamlit Cloud tempat menyimpan model
MODEL_DIR = "downloaded_models" # Nama folder lokal
INDOBERT_ZIP_PATH = os.path.join(MODEL_DIR, "final_indobert_model.zip")
INDOBERT_EXTRACT_PATH = os.path.join(MODEL_DIR, "final_indobert_model") # Path setelah diekstrak
LOGREG_TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_logreg_pipeline_final.joblib")
MNB_TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_mnb_pipeline_final.joblib")
META_MODEL_PATH = os.path.join(MODEL_DIR, "meta_model_final.joblib")
# SCALER_PATH = os.path.join(MODEL_DIR, "meta_feature_scaler.joblib") # Jika pakai scaler

# --- Konfigurasi Model (Sesuaikan jika berbeda dari training) ---
MAX_LEN_INDOBERT = 256
EVAL_BATCH_SIZE_INDOBERT = 32

# --- Fungsi Pra-pemrosesan Teks Dasar (Harus sama dengan saat training) ---
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\[(hoaks|salah|klarifikasi|disinformasi|fakta)\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    # text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Fungsi Download dari GDrive ---
def download_from_gdrive(file_id, output_path):
    """Download file dari Google Drive menggunakan gdown"""
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Downloading {os.path.basename(output_path)} from GDrive...")
    try:
        gdown.download(url, output_path, quiet=False)
        if os.path.exists(output_path):
             print(f"Successfully downloaded to {output_path}")
             return True
        else:
             print(f"Download failed, file not found: {output_path}")
             return False
    except Exception as e:
        print(f"Error downloading {output_path}: {e}")
        return False

# --- Fungsi Ekstrak ZIP ---
def extract_zip(zip_path, extract_to):
    """Ekstrak file ZIP"""
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully extracted.")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

# --- Fungsi untuk Memuat Semua Model (Gunakan Cache Resource) ---
@st.cache_resource # Cache resource agar model tidak di-load ulang setiap interaksi
def load_all_models():
    print("Attempting to load models...")
    models = {}
    # Buat direktori jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Download Semua Komponen ---
    download_ok = True
    # Download & Ekstrak IndoBERT
    if not os.path.isdir(INDOBERT_EXTRACT_PATH): # Hanya download jika belum ada
        if download_from_gdrive(GDRIVE_ID_INDOBERT_ZIP, INDOBERT_ZIP_PATH):
            if extract_zip(INDOBERT_ZIP_PATH, MODEL_DIR): # Ekstrak ke MODEL_DIR
                 # Hapus file zip setelah diekstrak untuk hemat space
                 try:
                     os.remove(INDOBERT_ZIP_PATH)
                     print(f"Removed {INDOBERT_ZIP_PATH}")
                 except OSError as e:
                     print(f"Error removing zip file: {e}")
            else:
                download_ok = False # Gagal ekstrak
        else:
            download_ok = False # Gagal download
    else:
        print(f"IndoBERT directory already exists: {INDOBERT_EXTRACT_PATH}")

    # Download model Sklearn & Meta
    if not os.path.exists(LOGREG_TFIDF_PATH):
        if not download_from_gdrive(GDRIVE_ID_LOGREG_TFIDF, LOGREG_TFIDF_PATH): download_ok = False
    if not os.path.exists(MNB_TFIDF_PATH):
        if not download_from_gdrive(GDRIVE_ID_MNB_TFIDF, MNB_TFIDF_PATH): download_ok = False
    if not os.path.exists(META_MODEL_PATH):
        if not download_from_gdrive(GDRIVE_ID_META_MODEL, META_MODEL_PATH): download_ok = False
    # Download scaler jika ada
    # if not os.path.exists(SCALER_PATH):
    #     if not download_from_gdrive(GDRIVE_ID_SCALER, SCALER_PATH): download_ok = False

    if not download_ok:
        st.error("Gagal mengunduh satu atau lebih komponen model dari Google Drive. Periksa ID File dan pengaturan sharing.")
        return None

    # --- Load Semua Komponen ---
    print("\nLoading downloaded models into memory...")
    try:
        models['tfidf_logreg'] = joblib.load(LOGREG_TFIDF_PATH)
        models['tfidf_mnb'] = joblib.load(MNB_TFIDF_PATH)
        models['meta_model'] = joblib.load(META_MODEL_PATH)
        # if os.path.exists(SCALER_PATH):
        #     models['scaler'] = joblib.load(SCALER_PATH)
        # else:
        #     models['scaler'] = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading IndoBERT on device: {device}")
        models['indobert_tokenizer'] = AutoTokenizer.from_pretrained(INDOBERT_EXTRACT_PATH)
        models['indobert_model'] = AutoModelForSequenceClassification.from_pretrained(INDOBERT_EXTRACT_PATH).to(device)
        models['indobert_model'].eval()
        models['device'] = device

        print("All models loaded successfully!")
        return models

    except Exception as e:
        st.error(f"Error loading models from local files: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# --- Fungsi Prediksi IndoBERT (Sama seperti sebelumnya) ---
def predict_indobert(_models, text_list):
    if not text_list or _models is None or 'indobert_model' not in _models:
        return np.array([[0.5, 0.5]] * len(text_list))

    tokenizer = _models['indobert_tokenizer']
    model = _models['indobert_model']
    device = _models['device']
    max_len = MAX_LEN_INDOBERT
    batch_size = EVAL_BATCH_SIZE_INDOBERT

    all_probabilities = []
    model.eval() # Pastikan mode eval
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i : i + batch_size]
            inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probabilities.extend(probabilities)

    return np.array(all_probabilities)

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Hoaks Indonesia",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Judul Aplikasi ---
st.title("🔍 Sistem Deteksi Hoaks Berita Indonesia (Stacking Ensemble)")
st.markdown("""
Masukkan **Judul** dan **Isi Narasi** berita dalam Bahasa Indonesia untuk dideteksi.
Aplikasi ini menggunakan *Stacking Ensemble* (IndoBERT, LogReg+TFIDF, MNB+TFIDF).
**Pastikan Anda telah memasukkan ID File Google Drive yang benar di kode.**
""")
st.divider()

# --- Load Model di Awal ---
if 'models' not in st.session_state:
    st.session_state.models = load_all_models()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Tampilkan status loading
if st.session_state.models is None:
    st.error("Gagal memuat model. Aplikasi tidak dapat berjalan.")
    st.stop() # Hentikan eksekusi jika model gagal load

# --- Input Pengguna ---
st.header("Masukkan Teks Berita")
if 'judul' not in st.session_state: st.session_state.judul = "Contoh: Beredar video Presiden menari di Istana"
if 'narasi' not in st.session_state: st.session_state.narasi = "Sebuah video yang diklaim menunjukkan Presiden sedang menari di Istana Negara baru-baru ini viral di media sosial. Video tersebut memperlihatkan sosok mirip Presiden melakukan gerakan tarian yang tidak biasa. Setelah ditelusuri, video tersebut adalah hasil editan."

judul = st.text_input("Judul Berita:", value=st.session_state.judul)
narasi = st.text_area("Isi Narasi Berita:", value=st.session_state.narasi, height=200)

st.session_state.judul = judul
st.session_state.narasi = narasi

detect_button = st.button("🔎 Deteksi Hoaks", type="primary")

st.divider()

# --- Proses Deteksi dan Hasil ---
if detect_button:
    if not judul or not narasi:
        st.warning("Mohon masukkan Judul dan Isi Narasi berita.")
    else:
        with st.spinner("Menganalisis teks dan melakukan prediksi..."):
            start_time = time.time()
            # 1. Preprocess
            input_text = narasi # Gunakan narasi saja, atau gabung: judul + " " + narasi
            cleaned_text = preprocess_text(input_text)
            cleaned_text_list = [cleaned_text]

            # 2. Prediksi Base Models
            base_predictions = {}
            models_loaded = st.session_state.models
            # Prediksi Sklearn Models
            for name in ['tfidf_logreg', 'tfidf_mnb']: # Sesuaikan jika model sklearn berubah
                if name in models_loaded:
                    try:
                        proba = models_loaded[name].predict_proba(cleaned_text_list)[:, 1]
                        base_predictions[name] = proba[0]
                    except Exception as e:
                        st.error(f"Error prediksi model {name}: {e}")
                        base_predictions[name] = 0.5
                else:
                     base_predictions[name] = 0.5 # Default jika model tidak ada

            # Prediksi IndoBERT
            try:
                proba_indobert = predict_indobert(models_loaded, cleaned_text_list)
                base_predictions['indoBERT'] = proba_indobert[0, 1]
            except Exception as e:
                 st.error(f"Error prediksi model IndoBERT: {e}")
                 base_predictions['indoBERT'] = 0.5

            # 3. Siapkan Meta-Features
            # !! PENTING: Urutan harus SAMA PERSIS dengan saat training meta-model !!
            feature_order = ['tfidf_logreg', 'tfidf_mnb', 'indoBERT'] # Sesuaikan urutan ini!
            try:
                # Pastikan semua prediksi base model ada sebelum membuat array
                if all(name in base_predictions for name in feature_order):
                     meta_features_input = np.array([[base_predictions[name] for name in feature_order]])
                     meta_features_input_scaled = meta_features_input # Asumsi tidak pakai scaler
                     # Jika pakai scaler:
                     # if models_loaded.get('scaler'):
                     #    meta_features_input_scaled = models_loaded['scaler'].transform(meta_features_input)

                     # 4. Prediksi Meta-Model
                     meta_model = models_loaded['meta_model']
                     final_proba = meta_model.predict_proba(meta_features_input_scaled)[:, 1]
                     final_pred = (final_proba >= 0.5).astype(int)[0]
                     confidence = final_proba[0] if final_pred == 1 else 1 - final_proba[0]

                     end_time = time.time()
                     process_time = end_time - start_time

                     # 5. Tampilkan Hasil
                     st.header("Hasil Deteksi")
                     col1, col2 = st.columns([1, 3])
                     with col1:
                         if final_pred == 1:
                             st.error(f"**HOAX**")
                             st.metric(label="Tingkat Keyakinan (Hoax)", value=f"{confidence:.1%}")
                         else:
                             st.success(f"**BUKAN HOAX**")
                             st.metric(label="Tingkat Keyakinan (Bukan Hoax)", value=f"{confidence:.1%}")
                         st.caption(f"Waktu proses: {process_time:.2f} detik")

                     with col2:
                         st.subheader("Detail Prediksi Base Models (Probabilitas Hoax)")
                         df_base_preds = pd.DataFrame([base_predictions])[feature_order] # Tampilkan sesuai urutan
                         st.dataframe(df_base_preds.round(4), use_container_width=True)
                         st.info("Interpretasi SHAP detail tidak ditampilkan di versi ini.")

                else:
                     st.error("Gagal mendapatkan prediksi dari semua base model untuk meta-features.")

            except KeyError as e:
                 st.error(f"Gagal membuat meta-feature. Pastikan urutan/nama base model benar: {e}")
            except Exception as e:
                 st.error(f"Error saat prediksi meta-model: {e}")
                 import traceback
                 st.code(traceback.format_exc())

# --- Footer ---
st.divider()
st.markdown("Aplikasi ini adalah prototipe dan hasil deteksi bersifat prediktif.")
st.markdown("Selalu rujuk ke sumber berita terpercaya.")

