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
import pickle
import requests
import io
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
    .loading-text {
        font-size: 1.2rem;
        color: #1976D2;
        text-align: center;
        margin: 2rem 0;
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

# Define the required classes for the model
class IndonesianTextPreprocessor:
    def __init__(self, remove_url=True, remove_html=True, remove_punctuation=True,
                 normalize_slang=True, remove_stopwords=True, stemming=True):
        self.remove_url = remove_url
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.normalize_slang = normalize_slang
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        
        # Initialize stopwords
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stopword_factory = StopWordRemoverFactory()
        self.stopwords_id = stopword_factory.get_stop_words()
        additional_stopwords = ['yg', 'dgn', 'nya', 'jd', 'klo', 'gak', 'ga', 'krn', 'nih', 'sih',
                            'jg', 'utk', 'tdk', 'sdh', 'dr', 'pd', 'dlm', 'tsb', 'tp', 'kk', 'ju',
                            'sy', 'jgn', 'ni', 'iy', 'bs', 'si', 'ya', 'lg', 'eh', 'kya', 'dah', 'loh', 'y', 'u']
        self.stopwords_id.extend(additional_stopwords)
        
        # Indonesian slang dictionary
        self.slang_words_dict = {
            'yg': 'yang', 'dgn': 'dengan', 'gak': 'tidak', 'ga': 'tidak', 'krn': 'karena',
            'udh': 'sudah', 'uda': 'sudah', 'udah': 'sudah', 'klo': 'kalau', 'gtu': 'begitu',
            'jd': 'jadi', 'jgn': 'jangan', 'bs': 'bisa', 'utk': 'untuk', 'u': 'kamu',
            'km': 'kamu', 'kmu': 'kamu', 'sy': 'saya', 'ak': 'aku', 'aq': 'aku',
            'tp': 'tapi', 'tdk': 'tidak', 'pd': 'pada', 'dl': 'dulu', 'dlu': 'dulu'
        }
        
        # Emoticon dictionary
        self.emoticon_dict = {
            ':)': 'senang', ':-)': 'senang', ':D': 'senang', ':-D': 'sangat_senang',
            ':(': 'sedih', ':-(': 'sedih', ':\'(': 'menangis', ':"(': 'menangis',
            ':p': 'bercanda', ':-p': 'bercanda', ':o': 'kaget', ':O': 'kaget'
        }
        
        # Initialize stemmer if needed
        if self.stemming:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        if self.remove_url:
            text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        if self.remove_html:
            text = re.sub(r'<.*?>', ' ', text)
        for emoticon, meaning in self.emoticon_dict.items():
            text = text.replace(emoticon, f' {meaning} ')
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        if self.normalize_slang:
            words = text.split()
            text = ' '.join([self.slang_words_dict.get(word, word) for word in words])
        tokens = word_tokenize(text)
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stopwords_id]
        if self.stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]
        text = ' '.join(tokens)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fit_transform(self, texts):
        processed_texts = []
        for text in texts:
            processed_texts.append(self.preprocess(text))
        return processed_texts


class SentimentFeatureExtractor:
    def __init__(self, extract_sentiment=True, extract_linguistic=True, extract_credibility=True):
        self.extract_sentiment = extract_sentiment
        self.extract_linguistic = extract_linguistic
        self.extract_credibility = extract_credibility
        self.feature_names = []
        
        # Sentiment dictionaries
        self.positive_words = ['baik', 'bagus', 'senang', 'gembira', 'indah', 'cantik', 'sukses', 'berhasil',
                          'setuju', 'benar', 'tepat', 'suka', 'cinta', 'sayang', 'peduli', 'terbaik',
                          'kuat', 'ramah', 'bijaksana', 'adil', 'jujur', 'damai', 'sempurna', 'hebat']
        self.negative_words = ['buruk', 'jelek', 'sedih', 'marah', 'benci', 'bodoh', 'gagal', 'salah',
                          'kecewa', 'susah', 'sulit', 'sakit', 'menderita', 'takut', 'cemas', 'khawatir']
        
        # Clickbait and hyperbole words
        self.clickbait_words = ['wow', 'gila', 'mengejutkan', 'mencengangkan', 'viral', 'terbongkar', 'rahasia']
        self.hyperbolic_words = ['sangat', 'sekali', 'terlalu', 'banget', 'maha', 'super', 'ultra']
        
        # Credibility indicators
        self.credibility_negative = ['hoax', 'bohong', 'palsu', 'tipu', 'menipu']
        self.credibility_positive = ['fakta', 'terbukti', 'resmi', 'otentik', 'valid']

    def _count_words(self, text, word_list):
        return sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text.lower())) for word in word_list)

    def _extract_sentiment_features(self, text):
        features = {}
        features['positive_count'] = self._count_words(text, self.positive_words)
        features['negative_count'] = self._count_words(text, self.negative_words)
        total = features['positive_count'] + features['negative_count']
        if total > 0:
            features['positive_ratio'] = features['positive_count'] / total
            features['negative_ratio'] = features['negative_count'] / total
            features['sentiment_score'] = (features['positive_count'] - features['negative_count']) / total
        else:
            features['positive_ratio'] = features['negative_ratio'] = features['sentiment_score'] = 0
        return features

    def _extract_linguistic_features(self, text):
        features = {}
        features['char_count'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        sentences = re.split(r'[.!?]+', text)
        features['sentence_count'] = sum(1 for s in sentences if s.strip())
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / features['char_count'] if features['char_count'] > 0 else 0
        features['all_caps_words'] = sum(1 for word in words if word.isupper() and len(word) > 1)
        return features

    def _extract_credibility_features(self, text):
        features = {}
        features['clickbait_count'] = self._count_words(text, self.clickbait_words)
        features['hyperbolic_count'] = self._count_words(text, self.hyperbolic_words)
        features['credibility_negative'] = self._count_words(text, self.credibility_negative)
        features['credibility_positive'] = self._count_words(text, self.credibility_positive)
        total_cred = features['credibility_negative'] + features['credibility_positive']
        features['credibility_score'] = (features['credibility_positive'] - features['credibility_negative']) / total_cred if total_cred > 0 else 0
        word_count = len(text.split())
        features['clickbait_ratio'] = features['clickbait_count'] / word_count if word_count > 0 else 0
        features['hyperbolic_ratio'] = features['hyperbolic_count'] / word_count if word_count > 0 else 0
        return features

    def extract_features(self, texts):
        feature_dict = []
        for text in texts:
            features = {}
            if self.extract_sentiment:
                features.update(self._extract_sentiment_features(text))
            if self.extract_linguistic:
                features.update(self._extract_linguistic_features(text))
            if self.extract_credibility:
                features.update(self._extract_credibility_features(text))
            feature_dict.append(features)
        if not self.feature_names and feature_dict:
            self.feature_names = list(feature_dict[0].keys())
        return pd.DataFrame(feature_dict)

    def fit_transform(self, texts):
        return self.extract_features(texts)


# Progress tracker (simplified for streamlit)
class TrainingProgressTracker:
    def __init__(self, total_models=0, verbose=True):
        self.verbose = verbose
        self.start_time = time.time()
        self.step_times = {}
        self.model_times = {}
        self.current_step = None
        self.total_models = total_models
        self.completed_models = 0

    def start_step(self, step_name):
        self.current_step = step_name
        self.step_times[step_name] = {"start": time.time()}

    def end_step(self, step_name, additional_info=None):
        if step_name in self.step_times:
            self.step_times[step_name]["end"] = time.time()
            self.step_times[step_name]["elapsed"] = self.step_times[step_name]["end"] - self.step_times[step_name]["start"]

    def get_summary(self):
        total_time = time.time() - self.start_time
        return {"total_training_time": total_time, "step_times": self.step_times, "model_times": self.model_times}


# Main HoaxDetectionSystem class (simplified for inference)
class HoaxDetectionSystem:
    def __init__(self, use_gpu=False, verbose=False, display_live_chart=False):
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.display_live_chart = display_live_chart
        
        self.preprocessor = IndonesianTextPreprocessor()
        self.feature_extractor = SentimentFeatureExtractor()
        
        # Initialize tokenizer and scaler (will be loaded with the model)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import MinMaxScaler
        self.tfidf_vectorizer = TfidfVectorizer()
        self.scaler = MinMaxScaler()
        
        self.base_models = {}
        self.ensemble_models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.is_trained = False
        self.class_balance = None
        self.tracker = TrainingProgressTracker(verbose=verbose)

    def _prepare_data(self, judul_series, narasi_series):
        """Prepare data for prediction"""
        self.tracker.start_step("Prediction data preparation")

        # Preprocess text
        preprocessed_judul = self.preprocessor.fit_transform(judul_series)
        preprocessed_narasi = self.preprocessor.fit_transform(narasi_series)

        # Apply TF-IDF
        judul_tfidf = self.tfidf_vectorizer.transform(preprocessed_judul)
        narasi_tfidf = self.tfidf_vectorizer.transform(preprocessed_narasi)

        # Extract other features
        judul_meta = self.feature_extractor.extract_features(judul_series).add_prefix('judul_')
        narasi_meta = self.feature_extractor.extract_features(narasi_series).add_prefix('narasi_')
        meta_features = pd.concat([judul_meta, narasi_meta], axis=1)

        # Scale features
        scaled = self.scaler.transform(meta_features)
        scaled_meta = pd.DataFrame(scaled, columns=meta_features.columns)

        # Combine features
        judul_cols = [f"judul_tfidf_{i}" for i in range(judul_tfidf.shape[1])]
        judul_df = pd.DataFrame(judul_tfidf.toarray(), columns=judul_cols)
        narasi_cols = [f"narasi_tfidf_{i}" for i in range(narasi_tfidf.shape[1])]
        narasi_df = pd.DataFrame(narasi_tfidf.toarray(), columns=narasi_cols)
        combined = pd.concat([judul_df, narasi_df, scaled_meta], axis=1)
        combined.columns = combined.columns.astype(str)

        # Handle missing columns
        for col in self.feature_names:
            if col not in combined.columns:
                combined[col] = 0

        # Ensure correct column order
        if self.feature_names:
            # Get intersection of columns
            common_columns = list(set(combined.columns).intersection(set(self.feature_names)))
            combined = combined[common_columns]

        self.tracker.end_step("Prediction data preparation")
        return combined

    def predict(self, data, text_column_judul='judul', text_column_narasi='narasi', return_proba=False, threshold=0.5):
        """Make prediction with the model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        X = self._prepare_data(data[text_column_judul], data[text_column_narasi])

        # Make prediction
        try:
            y_proba = self.best_model.predict_proba(X)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            if return_proba:
                return None, None
            return None

        if return_proba:
            return y_pred, y_proba
        return y_pred

    def explain_prediction(self, text_judul, text_narasi, num_features=10):
        """Explain the prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        input_df = pd.DataFrame({'judul': [text_judul], 'narasi': [text_narasi]})
        preprocessed_judul = self.preprocessor.preprocess(text_judul)
        preprocessed_narasi = self.preprocessor.preprocess(text_narasi)
        X = self._prepare_data(input_df['judul'], input_df['narasi'])

        prediction = self.best_model.predict(X)[0]
        probability = self.best_model.predict_proba(X)[0, 1]

        explanation = []

        # TF-IDF word importance analysis
        vectorizer = self.tfidf_vectorizer
        judul_tfidf = vectorizer.transform([preprocessed_judul])
        narasi_tfidf = vectorizer.transform([preprocessed_narasi])

        try:
            features = vectorizer.get_feature_names_out()
            judul_words = sorted([(features[idx], val) for idx, val in zip(judul_tfidf.indices, judul_tfidf.data)],
                                  key=lambda x: x[1], reverse=True)[:5]
            narasi_words = sorted([(features[idx], val) for idx, val in zip(narasi_tfidf.indices, narasi_tfidf.data)],
                                   key=lambda x: x[1], reverse=True)[:5]

            for word, val in judul_words:
                explanation.append({'feature': f'judul_{word}', 'shap_value': float(val),
                                  'direction': 'positive' if prediction==1 else 'negative'})
            for word, val in narasi_words:
                explanation.append({'feature': f'narasi_{word}', 'shap_value': float(val),
                                  'direction': 'positive' if prediction==1 else 'negative'})
        except Exception as e:
            st.warning(f"TF-IDF explanation error: {str(e)}")

        # Use feature importance from the model if available
        try:
            # Try to get feature importance
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
                feature_imp = feature_imp.sort_values('importance', ascending=False).head(num_features)
                
                for _, row in feature_imp.iterrows():
                    explanation.append({
                        'feature': row['feature'], 
                        'shap_value': float(row['importance']),
                        'direction': 'positive' if prediction==1 else 'negative'
                    })
            # For models like logistic regression
            elif hasattr(self.best_model, 'coef_'):
                importances = self.best_model.coef_[0]
                feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
                feature_imp = feature_imp.sort_values('importance', ascending=False).head(num_features)
                
                for _, row in feature_imp.iterrows():
                    explanation.append({
                        'feature': row['feature'], 
                        'shap_value': float(row['importance']),
                        'direction': 'positive' if row['importance'] > 0 else 'negative'
                    })
        except Exception as e:
            st.warning(f"Feature importance extraction error: {str(e)}")

        # Sort and limit by importance
        explanation = sorted(explanation, key=lambda x: abs(x['shap_value']), reverse=True)[:num_features]

        return {
            'prediction': int(prediction),
            'predicted_class': 'Hoax' if prediction==1 else 'Non-Hoax',
            'probability': probability,
            'confidence': probability if prediction==1 else 1-probability,
            'explanation': explanation
        }

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

# Function to load the model with caching
@st.cache_resource
def load_model(model_path):
    """Load the model with Streamlit caching"""
    try:
        st.info(f"Loading model from {model_path}...")
        start_time = time.time()
        
        model = joblib.load(model_path)
        
        # Initialize tracker if it's None
        if model.tracker is None:
            model.tracker = TrainingProgressTracker(verbose=False)
        
        # Set flags
        model.is_trained = True
        model.display_live_chart = False
        
        load_time = time.time() - start_time
        st.success(f"Model loaded successfully in {load_time:.2f} seconds!")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to test the app without model
def demo_prediction(judul, narasi, threshold=0.5):
    """Demo prediction function that works without the model"""
    # This is a placeholder function to simulate predictions without the actual model
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
    
    detector = None
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
            
            # Load the model
            detector = load_model("temp_model.pkl")
            if detector:
                model_loaded = True
    
    elif model_load_option == "Use Google Drive Link":
        gdrive_id = st.sidebar.text_input("Enter Google Drive file ID:", 
                                          help="Example: 1a2b3c4d5e - from the URL https://drive.google.com/file/d/1a2b3c4d5e/view")
        
        if gdrive_id and st.sidebar.button("Download & Load Model"):
            with st.spinner("Downloading model from Google Drive..."):
                download_success = download_model_from_gdrive(gdrive_id, "gdrive_model.pkl")
                
                if download_success:
                    detector = load_model("gdrive_model.pkl")
                    if detector:
                        model_loaded = True
                else:
                    st.sidebar.error("Failed to download the model from Google Drive")
    
    else:  # Use Local Path
        model_path = st.sidebar.text_input("Enter local model path:", value="model/hoax_detector_model.pkl")
        
        if model_path and st.sidebar.button("Load Model"):
            detector = load_model(model_path)
            if detector:
                model_loaded = True
    
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
            judul = st.text_area("Headline (Judul):", height=50,
                                value="Pemerintah Terapkan Protokol Kesehatan Baru untuk Atasi Covid-19")
            
            narasi = st.text_area("Content (Narasi):", height=150,
                                 value="Dalam upaya mengendalikan penyebaran Covid-19, pemerintah telah menerapkan protokol kesehatan yang lebih ketat, termasuk pembatasan pertemuan dan penerapan sanksi bagi pelanggar. Informasi ini berdasarkan data resmi dan rekomendasi dari ahli kesehatan.")
            
            threshold = st.slider("Detection Threshold:", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                                 help="Lower threshold makes the model more sensitive to hoaxes")
            
            submitted = st.form_submit_button("Detect Hoax")
        
        # Process the detection when form is submitted
        if submitted:
            # Display loading spinner
            with st.spinner("Analyzing content for hoax indicators..."):
                if demo_mode:
                    # Use demo prediction function
                    explanation = demo_prediction(judul, narasi, threshold)
                else:
                    # Create input DataFrame
                    input_df = pd.DataFrame({'judul': [judul], 'narasi': [narasi]})
                    
                    # Make prediction
                    pred, prob = detector.predict(input_df, return_proba=True, threshold=threshold)
                    
                    if pred is not None:
                        # Get explanation
                        explanation = detector.explain_prediction(judul, narasi)
                    else:
                        st.error("Error occurred during prediction. Using fallback demo mode.")
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
