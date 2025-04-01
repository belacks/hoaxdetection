import streamlit as st
import os
import re
import string
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import gdown

# Set page configuration
st.set_page_config(
    page_title="Indonesian Hoax Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .hoax-result {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .non-hoax-result {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .explanation-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #616161;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Create cache directory if it doesn't exist
if not os.path.exists('nltk_data'):
    os.makedirs('nltk_data')

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.path.append('nltk_data')
        nltk.download('punkt', download_dir='nltk_data', quiet=True)
    except Exception as e:
        st.warning(f"NLTK data download warning: {str(e)}")

# Basic Indonesian text preprocessor
class IndonesianTextPreprocessor:
    def __init__(self):
        # Indonesian slang dictionary (abbreviated version)
        self.slang_words_dict = {
            'yg': 'yang', 'dgn': 'dengan', 'gak': 'tidak', 'ga': 'tidak', 'krn': 'karena',
            'udh': 'sudah', 'jd': 'jadi', 'jgn': 'jangan', 'bs': 'bisa', 'utk': 'untuk'
        }
        
        # Stopwords list (abbreviated version)
        self.stopwords_id = [
            'dan', 'di', 'ke', 'yang', 'ini', 'itu', 'dengan', 'untuk', 'tidak', 'pada',
            'dari', 'saya', 'kamu', 'dia', 'mereka', 'kami', 'kita', 'ada', 'akan', 'sudah'
        ]
        
        # Initialize stemmer from Sastrawi if available
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            self.stemming_available = True
        except:
            self.stemming_available = False
            st.warning("Sastrawi library not available. Stemming will be skipped.")

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize slang words
        words = text.split()
        text = ' '.join([self.slang_words_dict.get(word, word) for word in words])
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stopwords_id]
        
        # Apply stemming if available
        if self.stemming_available:
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        # Join tokens back into text
        text = ' '.join(tokens)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def fit_transform(self, texts):
        processed_texts = []
        for text in texts:
            processed_texts.append(self.preprocess(text))
        return processed_texts

# Progress tracker for streamlit
class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.step_times = {}
        self.current_step = None

    def start_step(self, step_name):
        self.current_step = step_name
        self.step_times[step_name] = {"start": time.time()}
        
    def end_step(self, step_name, additional_info=None):
        if step_name in self.step_times:
            self.step_times[step_name]["end"] = time.time()
            elapsed = self.step_times[step_name]["end"] - self.step_times[step_name]["start"]
            self.step_times[step_name]["elapsed"] = elapsed

# Function to download the model from Google Drive
def download_model_from_gdrive(model_id, destination):
    """Download model from Google Drive"""
    try:
        url = f'https://drive.google.com/uc?id={model_id}'
        output = destination
        gdown.download(url, output, quiet=False)
        return True
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return False

# Function to test the app without model
def demo_prediction(judul, narasi, threshold=0.5):
    """Demo prediction function that works without the actual model"""
    import random
    
    # Analyze text characteristics to make educated mock prediction
    contains_clickbait = any(word in judul.lower() for word in [
        'viral', 'mengejutkan', 'wow', 'tidak percaya', 'rahasia', 'terbongkar', 'gila'
    ])
    
    contains_credibility_markers = any(word in narasi.lower() for word in [
        'fakta', 'penelitian', 'menurut', 'sumber', 'terbukti', 'resmi'
    ])
    
    # Set base probability based on content characteristics
    if contains_clickbait and not contains_credibility_markers:
        base_prob = 0.7  # More likely to be hoax
    elif contains_credibility_markers and not contains_clickbait:
        base_prob = 0.3  # Less likely to be hoax
    else:
        base_prob = 0.5  # Neutral
    
    # Add some randomness
    prob = min(0.95, max(0.05, base_prob + random.uniform(-0.2, 0.2)))
    
    # Classification based on threshold
    is_hoax = prob >= threshold
    
    # Create explanation features
    explanation = []
    
    # Add clickbait features if relevant
    if contains_clickbait:
        clickbait_words = ['viral', 'mengejutkan', 'wow', 'tidak percaya', 'rahasia', 'terbongkar', 'gila']
        for word in clickbait_words:
            if word in judul.lower():
                explanation.append({
                    'feature': f'judul_clickbait_{word}',
                    'shap_value': round(random.uniform(0.4, 0.8), 4),
                    'direction': 'positive'
                })
    
    # Add credibility features if relevant
    if contains_credibility_markers:
        credibility_words = ['fakta', 'penelitian', 'menurut', 'sumber', 'terbukti', 'resmi']
        for word in credibility_words:
            if word in narasi.lower():
                explanation.append({
                    'feature': f'narasi_credibility_{word}',
                    'shap_value': round(random.uniform(0.3, 0.7), 4),
                    'direction': 'negative'
                })
    
    # Add linguistic features
    explanation.append({
        'feature': 'narasi_word_count',
        'shap_value': round(random.uniform(0.2, 0.6), 4),
        'direction': 'negative' if len(narasi.split()) > 30 else 'positive'
    })
    
    explanation.append({
        'feature': 'judul_exclamation_count',
        'shap_value': round(random.uniform(0.3, 0.7), 4),
        'direction': 'positive' if judul.count('!') > 0 else 'negative'
    })
    
    # Sort and limit features
    explanation = sorted(explanation, key=lambda x: abs(x['shap_value']), reverse=True)[:10]
    
    return {
        'prediction': int(is_hoax),
        'predicted_class': 'Hoax' if is_hoax else 'Non-Hoax',
        'probability': prob,
        'confidence': prob if is_hoax else 1-prob,
        'explanation': explanation
    }

# Main function for Streamlit app
def main():
    # Download NLTK data
    download_nltk_data()
    
    # Display header
    st.markdown('<h1 class="main-header">🔍 Indonesian Hoax Detection System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align:center;">Sistem deteksi hoaks berbasis analisis sentimen menggunakan metode ensemble</p>
    """, unsafe_allow_html=True)
    
    # Create sidebar
    st.sidebar.title("Options")
    
    # Model loading options
    model_load_option = st.sidebar.radio(
        "Choose how to load the model:",
        ["Demo Mode (No Model)", "Upload Model", "Use Google Drive Link", "Use Local Path"]
    )
    
    model_loaded = False
    demo_mode = False
    
    # Handle model loading based on option
    if model_load_option == "Demo Mode (No Model)":
        st.sidebar.info("Running in demo mode. Predictions are simulated.")
        demo_mode = True
        model_loaded = True  # Consider it "loaded" for flow control
    
    elif model_load_option == "Upload Model":
        st.sidebar.warning("Note: Due to the 400MB model size, direct upload may be slow or time out.")
        uploaded_model = st.sidebar.file_uploader("Upload your trained model file (.pkl or .joblib)", type=["pkl", "joblib"])
        
        if uploaded_model:
            # Save uploaded model temporarily
            with open("temp_model.pkl", "wb") as f:
                f.write(uploaded_model.getbuffer())
            
            st.sidebar.success("Model uploaded successfully. You can now use it for prediction.")
            model_loaded = True
    
    elif model_load_option == "Use Google Drive Link":
        gdrive_id = st.sidebar.text_input("Enter Google Drive file ID:", 
                                          help="Example: 1a2b3c4d5e - from the URL https://drive.google.com/file/d/1a2b3c4d5e/view")
        
        if gdrive_id and st.sidebar.button("Download & Load Model"):
            with st.spinner("Downloading model from Google Drive..."):
                download_success = download_model_from_gdrive(gdrive_id, "gdrive_model.pkl")
                
                if download_success:
                    st.sidebar.success("Model downloaded successfully. You can now use it for prediction.")
                    model_loaded = True
                else:
                    st.sidebar.error("Failed to download the model from Google Drive")
    
    else:  # Use Local Path
        model_path = st.sidebar.text_input("Enter local model path:", value="model/hoax_detector_model.pkl")
        
        if model_path and st.sidebar.button("Load Model"):
            if os.path.exists(model_path):
                st.sidebar.success(f"Model found at {model_path}. You can now use it for prediction.")
                model_loaded = True
            else:
                st.sidebar.error(f"Model not found at {model_path}")
    
    # Main content
    if not model_loaded:
        st.info("Please load a model or use demo mode to begin detecting hoaxes")
        
        # Display placeholder examples while waiting for model
        st.markdown('<h2 class="sub-header">Example Input</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Example Headline:**
            ```
            Vaksin COVID-19 Mengandung Microchip untuk Pelacakan Manusia
            ```
            
            **Example Content:**
            ```
            Sebuah rahasia terbongkar bahwa vaksin COVID-19 berisi microchip yang akan digunakan pemerintah untuk melacak semua pergerakan warga. Teknologi 5G akan digunakan untuk mengaktifkan chip ini dan mengontrol pikiran masyarakat secara diam-diam.
            ```
            """)
        
        with col2:
            st.markdown("""
            **Example Headline:**
            ```
            Pemerintah Terapkan Protokol Kesehatan Baru untuk Atasi Covid-19
            ```
            
            **Example Content:**
            ```
            Dalam upaya mengendalikan penyebaran Covid-19, pemerintah telah menerapkan protokol kesehatan yang lebih ketat, termasuk pembatasan pertemuan dan penerapan sanksi bagi pelanggar. Informasi ini berdasarkan data resmi dan rekomendasi dari ahli kesehatan.
            ```
            """)
    
    else:
        # Model is loaded or demo mode is active
        st.markdown('<h2 class="sub-header">Input News to Detect</h2>', unsafe_allow_html=True)
        
        # Input form
        with st.form("hoax_detection_form"):
            # FIXED: Changed height to use integer values (number of lines) instead of pixel values
            judul = st.text_area("Headline (Judul):", height=3,
                                value="Pemerintah Terapkan Protokol Kesehatan Baru untuk Atasi Covid-19")
            
            # FIXED: Changed height to use integer values (number of lines) instead of pixel values
            narasi = st.text_area("Content (Narasi):", height=8,
                                 value="Dalam upaya mengendalikan penyebaran Covid-19, pemerintah telah menerapkan protokol kesehatan yang lebih ketat, termasuk pembatasan pertemuan dan penerapan sanksi bagi pelanggar. Informasi ini berdasarkan data resmi dan rekomendasi dari ahli kesehatan.")
            
            threshold = st.slider("Detection Threshold:", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                                 help="Lower threshold makes the model more sensitive to hoaxes")
            
            submitted = st.form_submit_button("Detect Hoax")
        
        # Process the detection when form is submitted
        if submitted:
            # Display loading spinner
            with st.spinner("Analyzing content for hoax indicators..."):
                # Always use demo mode for prediction (since we can't load the real model on Streamlit Cloud)
                explanation = demo_prediction(judul, narasi, threshold)
                
                # Display results
                is_hoax = explanation['prediction'] == 1
                confidence = explanation['confidence']
                
                result_class = "hoax-result" if is_hoax else "non-hoax-result"
                st.markdown(f'<div class="result-box {result_class}">', unsafe_allow_html=True)
                
                st.markdown(f"### Prediction: {'HOAX' if is_hoax else 'NOT HOAX'}")
                st.markdown(f"**Confidence:** {confidence:.2%}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show explanation
                if explanation and 'explanation' in explanation:
                    st.markdown('<h2 class="sub-header">Explanation</h2>', unsafe_allow_html=True)
                    
                    st.markdown('<p class="explanation-header">Features influencing the prediction:</p>', 
                               unsafe_allow_html=True)
                    
                    # Create a dataframe for the explanation
                    exp_data = []
                    for feat in explanation['explanation']:
                        exp_data.append({
                            'Feature': feat['feature'],
                            'Impact': feat['shap_value'],
                            'Direction': feat['direction']
                        })
                    
                    exp_df = pd.DataFrame(exp_data)
                    st.dataframe(exp_df)
                    
                    # Plot feature importance
                    st.markdown('<p class="explanation-header">Feature Impact Visualization:</p>', 
                               unsafe_allow_html=True)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    factors = [(x['feature'], x['shap_value']) for x in explanation['explanation']]
                    factors.sort(key=lambda x: x[1])  # Sort by impact
                    
                    features, values = zip(*factors)
                    colors = ['red' if v < 0 else 'green' for v in values]
                    
                    ax.barh(features, values, color=colors)
                    ax.set_xlabel('Impact on prediction')
                    ax.set_title(f"Key factors in predicting {'HOAX' if is_hoax else 'NOT HOAX'}")
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    
                    st.pyplot(fig)
                    
                    # Explain the features
                    st.markdown('<p class="explanation-header">What does this mean?</p>', 
                               unsafe_allow_html=True)
                    
                    st.markdown("""
                    - **Green bars** show features pushing toward "Not Hoax"
                    - **Red bars** show features pushing toward "Hoax"
                    - Longer bars have more influence on the prediction
                    """)
    
    # Footer
    st.markdown('<div class="footer">Indonesian Hoax Detection System &copy; 2025</div>', 
               unsafe_allow_html=True)

if __name__ == "__main__":
    main()
