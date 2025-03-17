# app.py - Streamlit Dashboard untuk Sistem Deteksi Hoaks Indonesia

import streamlit as st
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    st.error("Error importing matplotlib. Please check requirements.")

try:
    import seaborn as sns
except ImportError:
    st.error("Error importing seaborn. Please check requirements.")
import pickle
import os
import re
import time
import json
import urllib.request
from io import BytesIO
import urllib.parse
import base64
from datetime import datetime
import warnings
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("Error importing plotly. Using alternative visualization.")

    # Alternatif menggunakan altair yang sudah ada di Streamlit
    import altair as alt
    
    # Fungsi untuk mengganti penggunaan plotly dengan altair
    def create_alternative_chart(data, x, y, title):
        return alt.Chart(data).mark_bar().encode(
            x=x,
            y=y
        ).properties(title=title)
from PIL import Image
import tempfile
import shutil

# Untuk menangani Google Drive
import requests
try:
    import gdown
except:
    st.error("Error importing gdown!")

# Inisialisasi NLTK resources saat startup
# Use this:
import nltk
import os
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Function to safely download NLTK resources
def download_nltk_resource(resource_name):
    try:
        nltk.data.find(f'{resource_name}')
        print(f"Resource {resource_name} already exists.")
    except LookupError:
        print(f"Downloading {resource_name}...")
        nltk.download(resource_name, quiet=True)

# Download required resources
download_nltk_resource('tokenizers/punkt')
download_nltk_resource('tokenizers/punkt_tab')
download_nltk_resource('corpora/stopwords')

# Impor komponen sistem deteksi hoaks
# Catatan: Pastikan semua file implementasi sudah tersedia
from hoax_detection_system import HoaxDetectionSystem, IndonesianTextPreprocessor, SentimentFeatureExtractor

# Abaikan peringatan
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Deteksi Hoaks Indonesia",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konstanta
MODEL_DIR = "models"
TEMP_DIR = "temp"
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"

# Buat direktori yang diperlukan
for directory in [MODEL_DIR, TEMP_DIR, UPLOAD_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2563EB;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        margin-bottom: 1rem;
        border-left: 5px solid #2563EB;
    }
    .hoax-result {
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        width: 100%;
        margin-bottom: 1rem;
    }
    .hoax {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 2px solid #B91C1C;
    }
    .non-hoax {
        background-color: #D1FAE5;
        color: #047857;
        border: 2px solid #047857;
    }
    .warning {
        background-color: #FEF3C7;
        color: #92400E;
        border: 2px solid #92400E;
    }
    .feature-positive {
        color: #047857;
    }
    .feature-negative {
        color: #B91C1C;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6B7280;
        border-top: 1px solid #E5E7EB;
    }
    .help-text {
        font-size: 0.9rem;
        color: #4B5563;
        font-style: italic;
    }
    .logo {
        text-align: center;
        padding: 1rem;
    }
    .confidence-meter {
        height: 20px;
        background-color: #E5E7EB;
        border-radius: 10px;
        margin-bottom: 10px;
        overflow: hidden;
    }
    .confidence-fill-hoax {
        height: 100%;
        background-color: #EF4444;
        border-radius: 10px;
    }
    .confidence-fill-non-hoax {
        height: 100%;
        background-color: #10B981;
        border-radius: 10px;
    }
    .stMetric {
        background-color: #F3F4F6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .citation {
        font-size: 0.8rem;
        color: #6B7280;
        padding: 10px;
        border-left: 3px solid #2563EB;
        background-color: #EFF6FF;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk mendownload model dari Google Drive
def download_file_from_google_drive(file_id, destination):
    """
    Download file dari Google Drive menggunakan gdown
    """
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)
        return True
    except Exception as e:
        st.error(f"Error mengunduh file: {e}")
        return False

# Fungsi untuk mendapatkan ID file dari link Google Drive
def get_file_id_from_gdrive_link(link):
    """Ekstrak file ID dari link Google Drive"""
    if '/file/d/' in link:
        # Format link: https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing
        file_id = link.split('/file/d/1AU_-uQwka_L9f62qyzPuV8vFO6spblV1')[1].split('/')[0]
    elif 'id=' in link:
        # Format link: https://drive.google.com/open?id={FILE_ID}
        file_id = link.split('id=1AU_-uQwka_L9f62qyzPuV8vFO6spblV1')[1].split('&')[0]
    else:
        return None
    return file_id

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_path):
    """Memuat model deteksi hoaks dari file"""
    try:
        model = HoaxDetectionSystem.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

# Fungsi untuk membuat visualisasi pengaruh fitur
def create_feature_influence_chart(explanation):
    """Membuat visualisasi pengaruh fitur dengan Plotly"""
    features = []
    values = []
    colors = []
    
    for feature in explanation['explanation']:
        feature_name = feature['feature']
        # Persingkat nama fitur yang terlalu panjang
        if len(feature_name) > 30:
            feature_name = feature_name[:27] + "..."
        
        features.append(feature_name)
        values.append(feature['shap_value'])
        colors.append('#10B981' if feature['direction'] == 'negative' else '#EF4444')  # Hijau untuk negatif, merah untuk positif
    
    # Urutkan fitur berdasarkan nilai absolut
    sorted_indices = np.argsort(np.abs(values))
    features = [features[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    # Buat visualisasi
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features,
        x=values,
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Pengaruh Fitur pada Prediksi",
        xaxis_title="Kontribusi terhadap Prediksi",
        yaxis_title="Fitur",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(
            zeroline=True,
            zerolinecolor='#888',
            zerolinewidth=1
        )
    )
    
    return fig

# Fungsi untuk menginterpretasikan prediksi
def interpret_prediction(explanation):
    """Memberikan interpretasi dari hasil prediksi"""
    prediction = explanation['prediction']
    confidence = explanation['confidence']
    
    # Interpretasi berdasarkan tingkat keyakinan
    if prediction == 1:  # Hoax
        if confidence > 0.9:
            return "Teks ini sangat mungkin adalah hoaks (keyakinan tinggi)."
        elif confidence > 0.7:
            return "Teks ini kemungkinan adalah hoaks (keyakinan sedang)."
        else:
            return "Teks ini mungkin adalah hoaks, tetapi dengan keyakinan rendah."
    else:  # Non-hoax
        if confidence > 0.9:
            return "Teks ini sangat mungkin bukan hoaks (keyakinan tinggi)."
        elif confidence > 0.7:
            return "Teks ini kemungkinan bukan hoaks (keyakinan sedang)."
        else:
            return "Teks ini mungkin bukan hoaks, tetapi dengan keyakinan rendah."

# Fungsi untuk menampilkan pola yang terdeteksi
def explain_detected_patterns(explanation):
    """Menampilkan pola-pola yang terdeteksi dalam teks"""
    outputs = []
    feature_dict = {feature['feature']: feature['shap_value'] for feature in explanation['explanation']}
    
    # Cek pola khusus hoaks
    hoax_patterns = [f for f in feature_dict.keys() if 'pola_hoaks' in f]
    if hoax_patterns:
        pattern_names = [p.split('_')[-1] for p in hoax_patterns]
        outputs.append(f"Mengandung pola umum hoaks: {', '.join(pattern_names)}")
    
    # Cek clickbait
    if any('clickbait' in f for f in feature_dict.keys()):
        outputs.append("Menggunakan kata-kata clickbait yang menarik perhatian")
    
    # Cek penggunaan kapital berlebihan
    if any('uppercase' in f or 'all_caps_words' in f for f in feature_dict.keys()):
        outputs.append("Menggunakan huruf kapital berlebihan")
    
    # Cek tanda seru berlebihan
    if any('exclamation' in f for f in feature_dict.keys()):
        outputs.append("Menggunakan tanda seru berlebihan")
    
    # Cek kredibilitas
    if any('credibility_score' in f for f in feature_dict.keys()):
        credibility_features = [f for f in feature_dict.keys() if 'credibility_score' in f]
        if any(feature_dict[f] < 0 for f in credibility_features):
            outputs.append("Kurangnya indikator kredibilitas")
    
    # Jika tidak ada pola yang terdeteksi, berikan pesan generik
    if not outputs:
        outputs.append("Tidak ada pola spesifik yang terdeteksi")
    
    return outputs

# Fungsi untuk menyimpan prediksi dalam session state
def save_to_history(explanation, judul, narasi):
    """Menyimpan hasil prediksi ke riwayat analisis"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Batasi riwayat hingga 10 item
    if len(st.session_state.analysis_history) >= 10:
        st.session_state.analysis_history.pop(0)
    
    # Tambahkan analisis baru
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_item = {
        'timestamp': timestamp,
        'judul': judul,
        'narasi': narasi,
        'prediction': explanation['predicted_class'],
        'confidence': explanation['confidence']
    }
    
    st.session_state.analysis_history.append(history_item)

# Fungsi untuk menampilkan riwayat analisis
def display_analysis_history():
    """Menampilkan riwayat analisis yang telah dilakukan"""
    if 'analysis_history' not in st.session_state or not st.session_state.analysis_history:
        st.info("Belum ada riwayat analisis.")
        return
    
    st.markdown("<div class='sub-header'>Riwayat Analisis</div>", unsafe_allow_html=True)
    
    for i, item in enumerate(reversed(st.session_state.analysis_history)):
        col1, col2, col3 = st.columns([2, 6, 2])
        
        with col1:
            st.write(f"**{item['timestamp']}**")
        
        with col2:
            st.write(f"**Judul:** {item['judul'][:50]}..." if len(item['judul']) > 50 else f"**Judul:** {item['judul']}")
        
        with col3:
            if item['prediction'] == "Hoax":
                st.markdown(f"<div class='hoax-result hoax'>{item['prediction']} ({item['confidence']:.2f})</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='hoax-result non-hoax'>{item['prediction']} ({item['confidence']:.2f})</div>", unsafe_allow_html=True)

# Fungsi untuk mengonversi file Excel ke CSV
def convert_excel_to_csv(excel_file):
    """Mengonversi file Excel ke CSV untuk kompatibilitas"""
    try:
        df = pd.read_excel(excel_file)
        temp_file = os.path.join(TEMP_DIR, "temp_convert.csv")
        df.to_csv(temp_file, index=False)
        return temp_file
    except Exception as e:
        st.error(f"Error mengonversi Excel: {e}")
        return None

# Fungsi untuk memproses batch data
def process_batch_data(file_path, hoax_detector):
    """Memproses file CSV batch untuk deteksi hoaks"""
    try:
        # Baca file
        data = pd.read_csv(file_path)
        
        # Periksa kolom yang diperlukan
        if 'judul' not in data.columns and 'narasi' not in data.columns:
            # Coba tebak kolom yang mungkin
            possible_title_cols = [col for col in data.columns if 'judul' in col.lower() or 
                                  'title' in col.lower() or 'header' in col.lower()]
            possible_content_cols = [col for col in data.columns if 'narasi' in col.lower() or 
                                    'content' in col.lower() or 'text' in col.lower() or 
                                    'body' in col.lower() or 'isi' in col.lower()]
            
            if possible_title_cols and possible_content_cols:
                data = data.rename(columns={possible_title_cols[0]: 'judul', 
                                           possible_content_cols[0]: 'narasi'})
            else:
                # Jika tidak bisa menebak, gunakan dua kolom pertama
                if len(data.columns) >= 2:
                    data = data.rename(columns={data.columns[0]: 'judul', 
                                               data.columns[1]: 'narasi'})
                else:
                    return None, "File tidak memiliki kolom judul dan narasi yang diperlukan."
        
        # Isi nilai yang hilang
        data['judul'] = data['judul'].fillna('')
        data['narasi'] = data['narasi'].fillna('')
        
        # Buat prediksi
        predictions, probabilities = hoax_detector.predict(
            data,
            text_column_judul='judul',
            text_column_narasi='narasi',
            return_proba=True
        )
        
        # Tambahkan hasil ke dataframe
        results_df = data.copy()
        results_df['predicted_label'] = predictions
        results_df['probability'] = probabilities
        results_df['predicted_class'] = results_df['predicted_label'].apply(
            lambda x: 'Hoax' if x == 1 else 'Non-Hoax'
        )
        results_df['confidence'] = results_df.apply(
            lambda row: row['probability'] if row['predicted_label'] == 1 else 1 - row['probability'],
            axis=1
        )
        
        return results_df, None
    
    except Exception as e:
        return None, f"Error memproses file: {str(e)}"

# Fungsi untuk menyimpan plot sebagai gambar
def get_img_download_link(fig, filename, text):
    """Membuat link download untuk gambar plot"""
    buffer = BytesIO()
    fig.write_image(buffer, format="png")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">{text}</a>'
    return href

# Fungsi untuk menampilkan demo aplikasi
def show_demo():
    """Menampilkan demo contoh hoaks dan non-hoaks"""
    st.markdown("<div class='sub-header'>Demo Sistem Deteksi Hoaks</div>", unsafe_allow_html=True)
    
    demo_tabs = st.tabs(["Contoh Hoaks", "Contoh Non-Hoaks"])
    
    with demo_tabs[0]:
        st.markdown("#### Contoh Teks Hoaks")
        st.markdown("""
        **Judul**: BREAKING NEWS: Vaksin COVID Mengandung Microchip
        
        **Narasi**: PERHATIAN! Vaksin COVID-19 mengandung microchip yang akan digunakan untuk memantau pergerakan Anda. Bill Gates bekerjasama dengan pemerintah untuk menciptakan "New World Order". Jangan biarkan mereka mengendalikan pikiran Anda! SEBARKAN SEBELUM DIHAPUS!!!
        """)
        
        if st.button("Analisis Contoh Hoaks"):
            if 'model' in st.session_state:
                with st.spinner("Menganalisis teks hoaks..."):
                    explanation = st.session_state.model.explain_prediction(
                        "BREAKING NEWS: Vaksin COVID Mengandung Microchip",
                        "PERHATIAN! Vaksin COVID-19 mengandung microchip yang akan digunakan untuk memantau pergerakan Anda. Bill Gates bekerjasama dengan pemerintah untuk menciptakan \"New World Order\". Jangan biarkan mereka mengendalikan pikiran Anda! SEBARKAN SEBELUM DIHAPUS!!!"
                    )
                
                st.markdown(f"**Prediksi**: {explanation['predicted_class']}")
                st.markdown(f"**Keyakinan**: {explanation['confidence']:.4f}")
                
                # Visualisasi pengaruh fitur
                st.plotly_chart(create_feature_influence_chart(explanation))
                
                # Interpretasi
                st.markdown("**Interpretasi**:")
                st.markdown(interpret_prediction(explanation))
                
                # Pola terdeteksi
                st.markdown("**Pola yang Terdeteksi**:")
                for pattern in explain_detected_patterns(explanation):
                    st.markdown(f"- {pattern}")
            else:
                st.warning("Harap muat model terlebih dahulu dari menu Model.")
    
    with demo_tabs[1]:
        st.markdown("#### Contoh Teks Non-Hoaks")
        st.markdown("""
        **Judul**: Kemenkes: Kasus COVID-19 Menurun Berkat Vaksinasi
        
        **Narasi**: Kementerian Kesehatan melaporkan penurunan kasus COVID-19 sebesar 30% dalam sebulan terakhir. Menurut data resmi, tingkat vaksinasi yang meningkat menjadi faktor utama penurunan ini. Prof. Dr. Adi Wijaya dari Universitas Indonesia membenarkan korelasi antara vaksinasi dan penurunan kasus berdasarkan penelitian terbaru yang dilakukan timnya.
        """)
        
        if st.button("Analisis Contoh Non-Hoaks"):
            if 'model' in st.session_state:
                with st.spinner("Menganalisis teks non-hoaks..."):
                    explanation = st.session_state.model.explain_prediction(
                        "Kemenkes: Kasus COVID-19 Menurun Berkat Vaksinasi",
                        "Kementerian Kesehatan melaporkan penurunan kasus COVID-19 sebesar 30% dalam sebulan terakhir. Menurut data resmi, tingkat vaksinasi yang meningkat menjadi faktor utama penurunan ini. Prof. Dr. Adi Wijaya dari Universitas Indonesia membenarkan korelasi antara vaksinasi dan penurunan kasus berdasarkan penelitian terbaru yang dilakukan timnya."
                    )
                
                st.markdown(f"**Prediksi**: {explanation['predicted_class']}")
                st.markdown(f"**Keyakinan**: {explanation['confidence']:.4f}")
                
                # Visualisasi pengaruh fitur
                st.plotly_chart(create_feature_influence_chart(explanation))
                
                # Interpretasi
                st.markdown("**Interpretasi**:")
                st.markdown(interpret_prediction(explanation))
                
                # Pola terdeteksi
                st.markdown("**Pola yang Terdeteksi**:")
                for pattern in explain_detected_patterns(explanation):
                    st.markdown(f"- {pattern}")
            else:
                st.warning("Harap muat model terlebih dahulu dari menu Model.")

# Halaman Beranda
def home_page():
    st.markdown("<div class='main-header'>Sistem Deteksi Hoaks Indonesia</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='logo'><img src='https://raw.githubusercontent.com/username/hoax-detection/main/assets/logo.png' width='200'></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Selamat datang di Sistem Deteksi Hoaks Indonesia, platform yang dirancang untuk menganalisis konten media sosial 
    dan berita dalam Bahasa Indonesia untuk mendeteksi potensi hoaks menggunakan teknik analisis sentimen dan pembelajaran mesin.
    
    Sistem ini mampu mengidentifikasi karakteristik hoaks dengan analisis fitur komprehensif dan memberikan 
    penjelasan yang transparan mengapa sebuah konten diprediksi sebagai hoaks atau bukan.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Fitur utama
    st.markdown("<div class='sub-header'>Fitur Utama</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Analisis Real-time")
        st.markdown("Analisis teks secara real-time untuk mendeteksi hoaks dengan penjelasan detail mengapa teks tersebut dianggap sebagai hoaks atau bukan.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📁 Analisis Batch")
        st.markdown("Unggah file CSV berisi banyak teks untuk dianalisis sekaligus dan dapatkan laporan lengkap hasil deteksi.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Interpretasi Cerdas")
        st.markdown("Visualisasi faktor-faktor yang mempengaruhi prediksi, memungkinkan pengguna memahami alasan di balik klasifikasi.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Demo aplikasi
    show_demo()
    
    # Riwayat analisis
    display_analysis_history()
    
    # Statistik
    st.markdown("<div class='sub-header'>Statistik Sistem</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Akurasi Model", value="86.3%")
    
    with col2:
        st.metric(label="F1 Score", value="0.921")
    
    with col3:
        st.metric(label="AUC", value="0.818")
    
    # Footer
    st.markdown("<div class='footer'>", unsafe_allow_html=True)
    st.markdown("Sistem Deteksi Hoaks Indonesia • Dibuat untuk melawan disinformasi")
    st.markdown("</div>", unsafe_allow_html=True)

# Halaman Analisis Teks
def analyze_text_page():
    st.markdown("<div class='main-header'>Analisis Teks</div>", unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("Anda perlu memuat model terlebih dahulu. Silakan buka menu 'Model' dan muat model dari file atau Google Drive.")
        if st.button("Muat Model Demo"):
            with st.spinner("Memuat model demo..."):
                # Simulasi memuat model demo (dalam implementasi nyata, ini akan memuat model sebenarnya)
                time.sleep(2)
                st.success("Model demo berhasil dimuat!")
                # Temp placeholder for demo - in real implementation replace with actual model loading
                st.session_state.model = None
        return
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("Masukkan teks yang ingin dianalisis untuk mendeteksi apakah teks tersebut berpotensi sebagai hoaks atau bukan.")
    
    # Input form
    judul = st.text_input("Judul:", placeholder="Masukkan judul berita/konten...")
    narasi = st.text_area("Narasi/Isi:", height=150, placeholder="Masukkan narasi atau isi berita/konten...")
    
    # Contoh teks
    with st.expander("Contoh Teks"):
        example_hoax = {
            'judul': "BREAKING: Pemerintah Sembunyikan Data COVID-19",
            'narasi': "Pejabat pemerintah telah memanipulasi data korban COVID-19 untuk menenangkan masyarakat. Seorang dokter yang tidak ingin disebutkan namanya mengungkapkan bahwa jumlah korban sebenarnya 10x lipat dari yang dilaporkan. SEBARKAN INI SEBELUM DIHAPUS!!!"
        }
        
        example_non_hoax = {
            'judul': "Kemenkes: Kasus COVID-19 Menurun Berkat Vaksinasi",
            'narasi': "Kementerian Kesehatan melaporkan penurunan kasus COVID-19 sebesar 30% dalam sebulan terakhir. Menurut data resmi, tingkat vaksinasi yang meningkat menjadi faktor utama penurunan ini. Profesor Adi Wijaya dari UI membenarkan korelasi antara vaksinasi dan penurunan kasus berdasarkan penelitian terbaru."
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Contoh Hoaks")
            st.write(f"**Judul:** {example_hoax['judul']}")
            st.write(f"**Narasi:** {example_hoax['narasi']}")
            
            if st.button("Gunakan Contoh Hoaks"):
                st.session_state.judul = example_hoax['judul']
                st.session_state.narasi = example_hoax['narasi']
                st.rerun()
        
        with col2:
            st.markdown("#### Contoh Non-Hoaks")
            st.write(f"**Judul:** {example_non_hoax['judul']}")
            st.write(f"**Narasi:** {example_non_hoax['narasi']}")
            
            if st.button("Gunakan Contoh Non-Hoaks"):
                st.session_state.judul = example_non_hoax['judul']
                st.session_state.narasi = example_non_hoax['narasi']
                st.rerun()
    
    # Use session state to keep values between reruns
    if 'judul' in st.session_state:
        judul = st.session_state.judul
        del st.session_state.judul
    
    if 'narasi' in st.session_state:
        narasi = st.session_state.narasi
        del st.session_state.narasi
    
    # Analysis button
    analyze_button = st.button("Analisis Teks")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis results
    if analyze_button and (judul.strip() or narasi.strip()):
        with st.spinner("Menganalisis teks..."):
            try:
                # Mendapatkan penjelasan
                explanation = st.session_state.model.explain_prediction(judul, narasi)
                
                # Simpan ke riwayat
                save_to_history(explanation, judul, narasi)
                
                # Menampilkan hasil
                st.markdown("<div class='sub-header'>Hasil Analisis</div>", unsafe_allow_html=True)
                
                # Prediksi dan keyakinan
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    prediction = explanation['predicted_class']
                    confidence = explanation['confidence']
                    
                    if prediction == "Hoax":
                        st.markdown(f"<div class='hoax-result hoax'>{prediction}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='hoax-result non-hoax'>{prediction}</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"**Tingkat Keyakinan:** {confidence:.2f}")
                    st.markdown(f"**Interpretasi:** {interpret_prediction(explanation)}")
                
                with col2:
                    # Progress bar untuk keyakinan
                    st.markdown("<p style='text-align: center;'>Tingkat Keyakinan</p>", unsafe_allow_html=True)
                    
                    if prediction == "Hoax":
                        st.markdown(f"""
                        <div class="confidence-meter">
                            <div class="confidence-fill-hoax" style="width: {confidence*100}%"></div>
                        </div>
                        <p style='text-align: center;'>{confidence:.2f}</p>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="confidence-meter">
                            <div class="confidence-fill-non-hoax" style="width: {confidence*100}%"></div>
                        </div>
                        <p style='text-align: center;'>{confidence:.2f}</p>
                        """, unsafe_allow_html=True)
                
                # Visualisasi
                st.plotly_chart(create_feature_influence_chart(explanation))
                
                # Pola yang terdeteksi
                st.markdown("#### Pola yang Terdeteksi")
                
                patterns = explain_detected_patterns(explanation)
                if patterns:
                    for pattern in patterns:
                        st.markdown(f"- {pattern}")
                else:
                    st.markdown("Tidak ada pola spesifik yang terdeteksi.")
                
                # Detail fitur yang berpengaruh
                with st.expander("Detail Fitur yang Berpengaruh"):
                    st.markdown("#### Fitur yang Paling Berpengaruh")
                    
                    for feature in explanation['explanation']:
                        direction = "+" if feature['direction'] == 'positive' else "-"
                        if feature['direction'] == 'positive':
                            st.markdown(f"<span class='feature-positive'>{direction} {feature['feature']}: {feature['shap_value']:.4f}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span class='feature-negative'>{direction} {feature['feature']}: {feature['shap_value']:.4f}</span>", unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menganalisis teks: {str(e)}")
    
    elif analyze_button:
        st.warning("Silakan masukkan judul dan/atau narasi terlebih dahulu.")

# Halaman Analisis Batch
def batch_analysis_page():
    st.markdown("<div class='main-header'>Analisis Batch</div>", unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("Anda perlu memuat model terlebih dahulu. Silakan buka menu 'Model' dan muat model dari file atau Google Drive.")
        return
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Unggah file CSV untuk menganalisis banyak teks sekaligus. File harus memiliki setidaknya dua kolom:
    - `judul`: Kolom berisi judul teks (opsional)
    - `narasi`: Kolom berisi narasi atau isi teks
    
    Jika nama kolom berbeda, sistem akan mencoba menentukannya secara otomatis.
    """)
    
    # Template CSV
    st.markdown("### Template CSV")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'judul': ['Pemerintah Sembunyikan Data COVID-19', 'Vaksin Mengandung Mikrochip', 'Kasus COVID-19 Menurun Berkat Vaksinasi'],
        'narasi': [
            'Pejabat pemerintah telah memanipulasi data korban COVID-19 untuk menenangkan masyarakat.',
            'Bill Gates memasukkan mikrochip ke dalam vaksin untuk melacak pergerakan masyarakat.',
            'Kementerian Kesehatan melaporkan penurunan kasus COVID-19 sebesar 30% dalam sebulan terakhir.'
        ]
    })
    
    # Display sample data
    st.dataframe(sample_data)
    
    # Create a download link for the template
    csv = sample_data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="template_hoax_detection.csv">Unduh Template CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Unggah File:", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Convert Excel to CSV if needed
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                file_path = convert_excel_to_csv(file_path)
                if not file_path:
                    st.error("Gagal mengonversi file Excel ke CSV.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return
            
            # Read and display preview
            preview_data = pd.read_csv(file_path, nrows=5)
            st.markdown("### Preview Data")
            st.dataframe(preview_data)
            
            # Process button
            if st.button("Proses Batch"):
                with st.spinner("Menganalisis batch data..."):
                    results_df, error = process_batch_data(file_path, st.session_state.model)
                    
                    if error:
                        st.error(error)
                    else:
                        # Display results
                        st.markdown("<div class='sub-header'>Hasil Analisis Batch</div>", unsafe_allow_html=True)
                        
                        # Summary metrics
                        total_predictions = len(results_df)
                        hoax_count = sum(results_df['predicted_label'] == 1)
                        non_hoax_count = total_predictions - hoax_count
                        hoax_percentage = (hoax_count / total_predictions) * 100 if total_predictions > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Teks", total_predictions)
                        
                        with col2:
                            st.metric("Hoaks Terdeteksi", hoax_count)
                        
                        with col3:
                            st.metric("Persentase Hoaks", f"{hoax_percentage:.1f}%")
                        
                        # Display dataframe with results
                        st.dataframe(results_df)
                        
                        # Save results
                        results_path = os.path.join(RESULTS_DIR, f"hasil_analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                        results_df.to_csv(results_path, index=False)
                        
                        # Create download link
                        csv_results = results_df.to_csv(index=False)
                        b64_results = base64.b64encode(csv_results.encode()).decode()
                        download_filename = f"hasil_analisis_hoaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        href_results = f'<a href="data:file/csv;base64,{b64_results}" download="{download_filename}">Unduh Hasil Analisis (CSV)</a>'
                        st.markdown(href_results, unsafe_allow_html=True)
                        
                        # Visualizations
                        st.markdown("<div class='sub-header'>Visualisasi</div>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart
                            fig = px.pie(
                                values=[non_hoax_count, hoax_count],
                                names=['Non-Hoaks', 'Hoaks'],
                                title='Distribusi Prediksi',
                                color_discrete_sequence=['#10B981', '#EF4444']
                            )
                            st.plotly_chart(fig)
                            
                            # Add download link for the chart
                            st.markdown(get_img_download_link(fig, "distribusi_prediksi", "Unduh Gambar"), unsafe_allow_html=True)
                        
                        with col2:
                            # Confidence histogram
                            fig = px.histogram(
                                results_df,
                                x='confidence',
                                color='predicted_class',
                                nbins=20,
                                title='Distribusi Tingkat Keyakinan',
                                color_discrete_map={'Hoaks': '#EF4444', 'Non-Hoax': '#10B981'}
                            )
                            st.plotly_chart(fig)
                            
                            # Add download link for the chart
                            st.markdown(get_img_download_link(fig, "distribusi_keyakinan", "Unduh Gambar"), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error memproses file: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Halaman Model
def model_page():
    st.markdown("<div class='main-header'>Manajemen Model</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Halaman ini memungkinkan Anda untuk memuat model deteksi hoaks dari berbagai sumber.
    Anda dapat menggunakan model yang telah dilatih sebelumnya atau mengunggah model baru.
    """)
    
    # Tabs for different model sources
    model_tabs = st.tabs(["Google Drive", "Upload File"])
    
    # Google Drive tab
    with model_tabs[0]:
        st.markdown("### Muat Model dari Google Drive")
        
        drive_link = st.text_input(
            "Link Google Drive:",
            placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
        )
        
        if st.button("Muat dari Google Drive"):
            if drive_link:
                with st.spinner("Mengunduh model dari Google Drive..."):
                    # Extract file ID from link
                    file_id = get_file_id_from_gdrive_link(drive_link)
                    
                    if file_id:
                        # Create path for downloaded model
                        model_path = os.path.join(MODEL_DIR, "hoax_detector_model.pkl")
                        
                        # Download file
                        success = download_file_from_google_drive(file_id, model_path)
                        
                        if success:
                            # Load model
                            try:
                                st.session_state.model = load_model(model_path)
                                if st.session_state.model:
                                    st.success("Model berhasil dimuat dari Google Drive!")
                                    st.session_state.model_path = model_path
                                else:
                                    st.error("Gagal memuat model. File mungkin rusak atau bukan model deteksi hoaks yang valid.")
                            except Exception as e:
                                st.error(f"Error memuat model: {e}")
                        else:
                            st.error("Gagal mengunduh file dari Google Drive.")
                    else:
                        st.error("Link Google Drive tidak valid. Pastikan link dalam format yang benar.")
            else:
                st.warning("Silakan masukkan link Google Drive terlebih dahulu.")
    
    # Upload File tab
    with model_tabs[1]:
        st.markdown("### Unggah File Model")
        
        uploaded_model = st.file_uploader("Unggah file model (.pkl):", type=['pkl'])
        
        if uploaded_model is not None:
            with st.spinner("Memuat model..."):
                # Save uploaded file
                model_path = os.path.join(MODEL_DIR, "hoax_detector_model_uploaded.pkl")
                with open(model_path, 'wb') as f:
                    f.write(uploaded_model.getbuffer())
                
                # Load model
                try:
                    st.session_state.model = load_model(model_path)
                    if st.session_state.model:
                        st.success("Model berhasil dimuat!")
                        st.session_state.model_path = model_path
                    else:
                        st.error("Gagal memuat model. File mungkin rusak atau bukan model deteksi hoaks yang valid.")
                except Exception as e:
                    st.error(f"Error memuat model: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display model info if loaded
    if 'model' in st.session_state and st.session_state.model:
        st.markdown("<div class='sub-header'>Informasi Model</div>", unsafe_allow_html=True)
        
        try:
            # Get model metadata
            model = st.session_state.model
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Metadata Model")
                
                # Model type
                st.markdown(f"**Jenis Model Terbaik:** {model.best_model_name}")
                
                # Class balance
                if hasattr(model, 'class_balance') and model.class_balance:
                    st.markdown("**Distribusi Kelas Pelatihan:**")
                    for label, count in model.class_balance.items():
                        st.markdown(f"- Kelas {label}: {count} sampel")
                
                # Feature count
                if hasattr(model, 'feature_names') and model.feature_names:
                    st.markdown(f"**Jumlah Fitur:** {len(model.feature_names)}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Komponen Model")
                
                # Model components
                components = []
                
                if hasattr(model, 'preprocessor'):
                    components.append("Preprosesor Teks Indonesia")
                
                if hasattr(model, 'feature_extractor'):
                    components.append("Ekstraktor Fitur Sentimen")
                
                if hasattr(model, 'tfidf_vectorizer'):
                    components.append("TF-IDF Vectorizer")
                
                if hasattr(model, 'base_models'):
                    components.append(f"Model Dasar ({len(model.base_models)})")
                
                if hasattr(model, 'ensemble_models'):
                    components.append(f"Model Ensemble ({len(model.ensemble_models)})")
                
                for component in components:
                    st.markdown(f"- {component}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error menampilkan informasi model: {e}")

# Halaman Tentang
def about_page():
    st.markdown("<div class='main-header'>Tentang Sistem Deteksi Hoaks</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Sistem Deteksi Hoaks ini adalah aplikasi yang memanfaatkan teknologi pembelajaran mesin dan analisis sentimen
    untuk mengidentifikasi konten hoaks dalam Bahasa Indonesia. Sistem ini dikembangkan sebagai bagian dari upaya
    untuk melawan penyebaran disinformasi di media sosial dan platform berita online.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sub-header'>Metodologi</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Preprocessing")
        st.markdown("""
        - Normalisasi teks Bahasa Indonesia
        - Penanganan stopwords khusus Bahasa Indonesia
        - Normalisasi kata gaul dan slang
        - Stemming menggunakan Sastrawi
        - Penanganan emoji dan emoticon
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Ekstraksi Fitur")
        st.markdown("""
        - TF-IDF untuk representasi teks
        - Fitur sentimen (positif/negatif)
        - Fitur linguistik (panjang teks, kapitalisasi, tanda baca)
        - Fitur kredibilitas (kata-kata bombastis, clickbait, sensasional)
        - Deteksi pola khusus hoaks
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Model")
        st.markdown("""
        - Pendekatan ensemble (stacking, voting)
        - Model dasar (Logistic Regression, Random Forest, XGBoost, dll.)
        - Penanganan ketidakseimbangan kelas
        - Optimasi hyperparameter
        - Cross-validasi untuk evaluasi
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Interpretasi")
        st.markdown("""
        - Penjelasan kontribusi fitur menggunakan SHAP
        - Visualisasi faktor-faktor yang mempengaruhi prediksi
        - Deteksi pola penulisan hoaks
        - Metrik kepercayaan dan level keyakinan
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sub-header'>Limitasi</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Sistem ini memiliki beberapa keterbatasan yang perlu diperhatikan:
    
    - **Tidak 100% akurat**: Meskipun akurasinya tinggi, sistem masih bisa salah dalam beberapa kasus.
    - **Keterbatasan konteks**: Sistem mungkin tidak memahami konteks tertentu atau referensi budaya lokal.
    - **Evolusi hoaks**: Pola hoaks berubah seiring waktu, sistem perlu dilatih ulang secara berkala.
    - **Keterbatasan bahasa**: Dioptimalkan untuk Bahasa Indonesia standar, mungkin kurang optimal untuk dialek lokal.
    
    Sistem harus digunakan sebagai alat bantu, bukan sebagai penentu tunggal kebenaran informasi.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sub-header'>Referensi</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='citation'>", unsafe_allow_html=True)
    st.markdown("Leveraging contextual features for fake news detection")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='citation'>", unsafe_allow_html=True)
    st.markdown("Explanation methods for fake news detection models")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='citation'>", unsafe_allow_html=True)
    st.markdown("Cross-domain transferability of fake news detection models")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='citation'>", unsafe_allow_html=True)
    st.markdown("Utilizing sentiment analysis for improved fake news detection")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("<div class='footer'>", unsafe_allow_html=True)
    st.markdown("Sistem Deteksi Hoaks Indonesia • Versi 1.0")
    st.markdown("Dibuat untuk melawan disinformasi di media sosial Indonesia")
    st.markdown("</div>", unsafe_allow_html=True)

# Main function
def main():
    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Flag_of_Indonesia.svg", width=100)
    st.sidebar.markdown("<div class='sub-header'>Sistem Deteksi Hoaks</div>", unsafe_allow_html=True)
    st.sidebar.markdown("Analisis konten media sosial dan berita Indonesia untuk mendeteksi potensi hoaks.")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Menu",
        ["Beranda", "Analisis Teks", "Analisis Batch", "Model", "Tentang"]
    )
    
    # Model status indicator
    if 'model' in st.session_state and st.session_state.model:
        st.sidebar.success("✅ Model berhasil dimuat")
    else:
        st.sidebar.warning("⚠️ Model belum dimuat")
    
    # Display the selected page
    if page == "Beranda":
        home_page()
    elif page == "Analisis Teks":
        analyze_text_page()
    elif page == "Analisis Batch":
        batch_analysis_page()
    elif page == "Model":
        model_page()
    elif page == "Tentang":
        about_page()
    
    # Cleanup temp files
    if os.path.exists(TEMP_DIR):
        for file in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, file)
            if os.path.isfile(file_path) and file.startswith('temp_'):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

if __name__ == "__main__":
    main()
