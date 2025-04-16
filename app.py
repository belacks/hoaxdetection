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

# --- Konfigurasi ---
# Sesuaikan path ini ke tempat Anda menyimpan model RELATIF terhadap app.py
# Misalnya, jika app.py ada di root, dan model di subfolder 'models'
MODEL_DIR = "models/"
INDOBERT_PATH = os.path.join(MODEL_DIR, "final_indobert_model")
LOGREG_TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_logreg_pipeline_final.joblib")
MNB_TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_mnb_pipeline_final.joblib")
META_MODEL_PATH = os.path.join(MODEL_DIR, "meta_model_final.joblib")
# SCALER_PATH = os.path.join(MODEL_DIR, "meta_feature_scaler.joblib") # Jika pakai scaler

# --- Fungsi Pra-pemrosesan Teks Dasar (Harus sama dengan saat training) ---
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\[(hoaks|salah|klarifikasi|disinformasi|fakta)\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    # text = re.sub(r'\d+', ' ', text) # Hapus angka jika dilakukan saat training
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Tambahkan aturan regex lain jika ada saat training
    return text

# --- Fungsi untuk Memuat Semua Model (Gunakan Cache Resource) ---
@st.cache_resource # Cache resource agar model tidak di-load ulang setiap interaksi
def load_all_models():
    print("Attempting to load models...")
    models = {}
    try:
        # Load Sklearn Pipelines
        if os.path.exists(LOGREG_TFIDF_PATH):
            models['tfidf_logreg'] = joblib.load(LOGREG_TFIDF_PATH)
            print("Loaded tfidf_logreg pipeline.")
        else:
            st.error(f"Model file not found: {LOGREG_TFIDF_PATH}")
            return None

        if os.path.exists(MNB_TFIDF_PATH):
            models['tfidf_mnb'] = joblib.load(MNB_TFIDF_PATH)
            print("Loaded tfidf_mnb pipeline.")
        else:
            st.error(f"Model file not found: {MNB_TFIDF_PATH}")
            return None
        # Load model sklearn lain jika ada (misal LGBM)

        # Load Meta-Model
        if os.path.exists(META_MODEL_PATH):
            models['meta_model'] = joblib.load(META_MODEL_PATH)
            print("Loaded meta_model.")
        else:
            st.error(f"Model file not found: {META_MODEL_PATH}")
            return None

        # Load IndoBERT Model & Tokenizer
        if os.path.isdir(INDOBERT_PATH):
            # Tentukan device (utamakan CUDA jika tersedia di Streamlit Cloud/env)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading IndoBERT on device: {device}")
            models['indobert_tokenizer'] = AutoTokenizer.from_pretrained(INDOBERT_PATH)
            models['indobert_model'] = AutoModelForSequenceClassification.from_pretrained(INDOBERT_PATH).to(device)
            models['indobert_model'].eval() # Set ke mode evaluasi
            models['device'] = device # Simpan device
            print("Loaded IndoBERT model and tokenizer.")
        else:
            st.error(f"IndoBERT model directory not found: {INDOBERT_PATH}")
            return None

        # Load Scaler (jika digunakan saat training meta-model)
        # if os.path.exists(SCALER_PATH):
        #     models['scaler'] = joblib.load(SCALER_PATH)
        #     print("Loaded meta-feature scaler.")
        # else:
        #     print("Meta-feature scaler not found, assuming no scaling needed.")
        #     models['scaler'] = None # Tandai tidak ada scaler

        print("All models loaded successfully!")
        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# --- Fungsi Prediksi IndoBERT ---
# Dibuat terpisah agar bisa di-cache jika input sama (meski sulit untuk teks bebas)
# @st.cache_data # Cache data mungkin kurang efektif untuk teks bebas
def predict_indobert(_models, text_list):
    """Membuat prediksi probabilitas menggunakan IndoBERT"""
    if not text_list or _models is None or 'indobert_model' not in _models:
        return np.array([[0.5, 0.5]] * len(text_list)) # Return neutral if error/not loaded

    tokenizer = _models['indobert_tokenizer']
    model = _models['indobert_model']
    device = _models['device']
    max_len = MAX_LEN_INDOBERT # Ambil dari konfigurasi
    batch_size = EVAL_BATCH_SIZE_INDOBERT # Gunakan batch eval

    all_probabilities = []
    with torch.no_grad(): # Penting untuk inference
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
st.title("🔍 Sistem Deteksi Hoaks Berita Indonesia")
st.markdown("""
Masukkan **Judul** dan **Isi Narasi** berita dalam Bahasa Indonesia untuk dideteksi.
Aplikasi ini menggunakan metode *Stacking Ensemble* yang menggabungkan beberapa model
Machine Learning dan Deep Learning (IndoBERT).
""")
st.divider()

# --- Load Model di Awal ---
# Hanya load sekali saat aplikasi dimulai atau jika belum ada di state
if 'models' not in st.session_state:
    with st.spinner("Memuat model... Proses ini mungkin perlu beberapa saat."):
        st.session_state.models = load_all_models()
        # Bersihkan memori setelah loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- Input Pengguna ---
st.header("Masukkan Teks Berita")

# Gunakan session state untuk menyimpan input terakhir
if 'judul' not in st.session_state: st.session_state.judul = "Contoh: Beredar video Presiden menari di Istana"
if 'narasi' not in st.session_state: st.session_state.narasi = "Sebuah video yang diklaim menunjukkan Presiden sedang menari di Istana Negara baru-baru ini viral di media sosial. Video tersebut memperlihatkan sosok mirip Presiden melakukan gerakan tarian yang tidak biasa. Setelah ditelusuri, video tersebut adalah hasil editan."

judul = st.text_input("Judul Berita:", value=st.session_state.judul)
narasi = st.text_area("Isi Narasi Berita:", value=st.session_state.narasi, height=200)

# Simpan input ke session state agar tidak hilang saat tombol ditekan
st.session_state.judul = judul
st.session_state.narasi = narasi

# Tombol Deteksi
detect_button = st.button("🔎 Deteksi Hoaks", type="primary")

st.divider()

# --- Proses Deteksi dan Hasil ---
if detect_button:
    if st.session_state.models is None:
        st.error("Model gagal dimuat. Silakan refresh halaman atau periksa file model.")
    elif not judul or not narasi:
        st.warning("Mohon masukkan Judul dan Isi Narasi berita.")
    else:
        with st.spinner("Menganalisis teks dan melakukan prediksi..."):
            start_time = time.time()

            # 1. Preprocess Input Text
            # Gabungkan judul dan narasi jika model dilatih demikian,
            # atau gunakan narasi saja jika itu input utama. Sesuaikan!
            # Diasumsikan model dilatih pada gabungan atau narasi saja.
            # Jika narasi adalah input utama:
            input_text = narasi
            # Jika gabungan:
            # input_text = judul + " " + narasi
            cleaned_text = preprocess_text(input_text)
            cleaned_text_list = [cleaned_text] # Buat jadi list untuk input model

            # 2. Prediksi Base Models
            base_predictions = {}
            # Prediksi Sklearn Models
            for name, model_pipeline in st.session_state.models.items():
                if name.startswith('tfidf_'): # Identifikasi pipeline sklearn
                    try:
                        # Pastikan input adalah list string
                        proba = model_pipeline.predict_proba(cleaned_text_list)[:, 1] # Ambil prob kelas 1 (Hoax)
                        base_predictions[name] = proba[0]
                    except Exception as e:
                        st.error(f"Error prediksi model {name}: {e}")
                        base_predictions[name] = 0.5 # Default jika error

            # Prediksi IndoBERT
            try:
                proba_indobert = predict_indobert(st.session_state.models, cleaned_text_list)
                base_predictions['indoBERT'] = proba_indobert[0, 1] # Ambil prob kelas 1 (Hoax)
            except Exception as e:
                 st.error(f"Error prediksi model IndoBERT: {e}")
                 base_predictions['indoBERT'] = 0.5 # Default jika error

            # 3. Siapkan Meta-Features
            # Pastikan urutan kolom SAMA seperti saat training meta-model
            # Urutan diasumsikan: 'tfidf_logreg', 'tfidf_mnb', 'indoBERT' (sesuaikan!)
            feature_order = ['tfidf_logreg', 'tfidf_mnb', 'indoBERT'] # SESUAIKAN URUTAN INI
            try:
                meta_features_input = np.array([[base_predictions[name] for name in feature_order]])
            except KeyError as e:
                 st.error(f"Gagal membuat meta-feature. Base model '{e}' tidak ditemukan atau gagal prediksi.")
                 meta_features_input = None

            if meta_features_input is not None:
                # 4. (Opsional) Scaling Meta-Features (jika scaler ada)
                # if st.session_state.models.get('scaler'):
                #     try:
                #         meta_features_input_scaled = st.session_state.models['scaler'].transform(meta_features_input)
                #     except Exception as e:
                #         st.error(f"Error scaling meta-features: {e}")
                #         meta_features_input_scaled = meta_features_input # Gunakan unscaled jika error
                # else:
                #     meta_features_input_scaled = meta_features_input # Langsung gunakan jika tidak ada scaler
                # Untuk contoh ini, kita asumsikan tidak pakai scaler
                meta_features_input_scaled = meta_features_input

                # 5. Prediksi Meta-Model
                try:
                    meta_model = st.session_state.models['meta_model']
                    final_proba = meta_model.predict_proba(meta_features_input_scaled)[:, 1] # Probabilitas Hoax
                    final_pred = (final_proba >= 0.5).astype(int)[0] # Threshold default 0.5
                    confidence = final_proba[0] if final_pred == 1 else 1 - final_proba[0]

                    end_time = time.time()
                    process_time = end_time - start_time

                    # 6. Tampilkan Hasil
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
                        # Buat dataframe dari base_predictions untuk tampilan lebih rapi
                        df_base_preds = pd.DataFrame([base_predictions])
                        st.dataframe(df_base_preds.round(4), use_container_width=True)
                        st.info("Interpretasi SHAP detail tidak ditampilkan di versi ini karena intensif komputasi.")

                except Exception as e:
                    st.error(f"Error saat prediksi meta-model: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# --- Footer ---
st.divider()
st.markdown("Aplikasi ini adalah prototipe dan hasil deteksi bersifat prediktif berdasarkan model yang dilatih.")
st.markdown("Untuk kepastian, selalu rujuk ke sumber berita terpercaya dan lakukan pengecekan fakta independen.")

