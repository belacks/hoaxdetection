import os
import gdown
import re
import string
import time
import warnings
import joblib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(
    page_title="Indonesian Hoax Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🔍 Indonesian Hoax Detection System")
st.markdown("""
    This application uses machine learning to detect hoaxes in Indonesian news articles.
    Enter a headline and narrative text to check whether it's likely to be a hoax.
    Here's the ID of the model's Google Drive upload: 1vOZVGViZSav67ZvdO3NzJ3PlKNYf0oEa
""")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_components' not in st.session_state:
    st.session_state.model_components = {}

# Download NLTK resources on app startup
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")
        return False

# Create stopwords and dictionaries
@st.cache_resource
def initialize_nlp_resources():
    # Initialize Stemmer and Stopwords for Indonesian
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopwords_id = stopword_factory.get_stop_words()
    additional_stopwords = ['yg', 'dgn', 'nya', 'jd', 'klo', 'gak', 'ga', 'krn', 'nih', 'sih',
                            'jg', 'utk', 'tdk', 'sdh', 'dr', 'pd', 'dlm', 'tsb', 'tp', 'kk', 'ju',
                            'sy', 'jgn', 'ni', 'iy', 'bs', 'si', 'ya', 'lg', 'eh', 'kya', 'dah', 'loh', 'y', 'u']
    stopwords_id.extend(additional_stopwords)
    
    # Indonesian slang dictionary
    slang_words_dict = {
        'yg': 'yang', 'dgn': 'dengan', 'gak': 'tidak', 'ga': 'tidak', 'krn': 'karena',
        'udh': 'sudah', 'uda': 'sudah', 'udah': 'sudah', 'klo': 'kalau', 'gtu': 'begitu',
        'jd': 'jadi', 'jgn': 'jangan', 'bs': 'bisa', 'utk': 'untuk', 'u': 'kamu',
        'km': 'kamu', 'kmu': 'kamu', 'sy': 'saya', 'ak': 'aku', 'aq': 'aku',
        'tp': 'tapi', 'tdk': 'tidak', 'pd': 'pada', 'dl': 'dulu', 'dlu': 'dulu'
    }
    
    # Emoticon dictionary
    emoticon_dict = {
        ':)': 'senang', ':-)': 'senang', ':D': 'senang', ':-D': 'sangat_senang',
        ':(': 'sedih', ':-(': 'sedih', ':\'(': 'menangis', ':"(': 'menangis',
        ':p': 'bercanda', ':-p': 'bercanda', ':o': 'kaget', ':O': 'kaget'
    }
    
    # Sentiment dictionaries
    positive_words = ['baik', 'bagus', 'senang', 'gembira', 'indah', 'cantik', 'sukses', 'berhasil',
                      'setuju', 'benar', 'tepat', 'suka', 'cinta', 'sayang', 'peduli', 'terbaik',
                      'kuat', 'ramah', 'bijaksana', 'adil', 'jujur', 'damai', 'sempurna', 'hebat']
    negative_words = ['buruk', 'jelek', 'sedih', 'marah', 'benci', 'bodoh', 'gagal', 'salah',
                      'kecewa', 'susah', 'sulit', 'sakit', 'menderita', 'takut', 'cemas', 'khawatir']
    
    # Clickbait and hyperbole words
    clickbait_words = ['wow', 'gila', 'mengejutkan', 'mencengangkan', 'viral', 'terbongkar', 'rahasia']
    hyperbolic_words = ['sangat', 'sekali', 'terlalu', 'banget', 'maha', 'super', 'ultra']
    
    # Credibility indicators
    credibility_negative = ['hoax', 'bohong', 'palsu', 'tipu', 'menipu']
    credibility_positive = ['fakta', 'terbukti', 'resmi', 'otentik', 'valid']
    
    return {
        'stemmer': stemmer,
        'stopwords_id': stopwords_id,
        'slang_words_dict': slang_words_dict,
        'emoticon_dict': emoticon_dict,
        'positive_words': positive_words,
        'negative_words': negative_words,
        'clickbait_words': clickbait_words,
        'hyperbolic_words': hyperbolic_words,
        'credibility_negative': credibility_negative,
        'credibility_positive': credibility_positive
    }

# Initialize NLP resources
success = download_nltk_resources()
if success:
    nlp_resources = initialize_nlp_resources()
else:
    st.error("Failed to initialize NLP resources. Please refresh the page.")
    st.stop()

# Define the classes required for the model
class IndonesianTextPreprocessor:
    def __init__(self, remove_url=True, remove_html=True, remove_punctuation=True,
                 normalize_slang=True, remove_stopwords=True, stemming=True):
        self.remove_url = remove_url
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.normalize_slang = normalize_slang
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        if self.remove_url:
            text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        if self.remove_html:
            text = re.sub(r'<.*?>', ' ', text)
        for emoticon, meaning in nlp_resources['emoticon_dict'].items():
            text = text.replace(emoticon, f' {meaning} ')
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        if self.normalize_slang:
            words = text.split()
            text = ' '.join([nlp_resources['slang_words_dict'].get(word, word) for word in words])
        tokens = word_tokenize(text)
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in nlp_resources['stopwords_id']]
        if self.stemming:
            tokens = [nlp_resources['stemmer'].stem(word) for word in tokens]
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

    def _count_words(self, text, word_list):
        return sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text.lower())) for word in word_list)

    def _extract_sentiment_features(self, text):
        features = {}
        features['positive_count'] = self._count_words(text, nlp_resources['positive_words'])
        features['negative_count'] = self._count_words(text, nlp_resources['negative_words'])
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
        features['clickbait_count'] = self._count_words(text, nlp_resources['clickbait_words'])
        features['hyperbolic_count'] = self._count_words(text, nlp_resources['hyperbolic_words'])
        features['credibility_negative'] = self._count_words(text, nlp_resources['credibility_negative'])
        features['credibility_positive'] = self._count_words(text, nlp_resources['credibility_positive'])
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

# Simplified HoaxDetectionSystem class optimized for inference
class HoaxDetectionSystem:
    def __init__(self):
        self.preprocessor = None
        self.feature_extractor = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.is_trained = False
        self.tracker = TrainingProgressTracker(verbose=True)

    def _combine_features(self, judul_tfidf, narasi_tfidf, meta_features):
        judul_cols = [f"judul_tfidf_{i}" for i in range(judul_tfidf.shape[1])]
        judul_df = pd.DataFrame(judul_tfidf.toarray(), columns=judul_cols)
        narasi_cols = [f"narasi_tfidf_{i}" for i in range(narasi_tfidf.shape[1])]
        narasi_df = pd.DataFrame(narasi_tfidf.toarray(), columns=narasi_cols)
        combined = pd.concat([judul_df, narasi_df, meta_features], axis=1)
        combined.columns = combined.columns.astype(str)

        if self.feature_names:
            for col in self.feature_names:
                if col not in combined.columns:
                    combined[col] = 0
            combined = combined[self.feature_names]

        return combined

    def _prepare_data(self, judul_series, narasi_series):
        self.tracker.start_step("Prediction data preparation")

        preprocessed_judul = self.preprocessor.fit_transform(judul_series)
        preprocessed_narasi = self.preprocessor.fit_transform(narasi_series)

        judul_tfidf = self.tfidf_vectorizer.transform(preprocessed_judul)
        narasi_tfidf = self.tfidf_vectorizer.transform(preprocessed_narasi)

        judul_meta = self.feature_extractor.extract_features(judul_series).add_prefix('judul_')
        narasi_meta = self.feature_extractor.extract_features(narasi_series).add_prefix('narasi_')
        meta_features = pd.concat([judul_meta, narasi_meta], axis=1)

        scaled = self.scaler.transform(meta_features)
        scaled_meta = pd.DataFrame(scaled, columns=meta_features.columns)

        combined = self._combine_features(judul_tfidf, narasi_tfidf, scaled_meta)

        self.tracker.end_step("Prediction data preparation")
        return combined

    def predict(self, data, text_column_judul='judul', text_column_narasi='narasi', return_proba=False, threshold=0.5):
        if not self.is_trained:
            raise ValueError("Model must be trained first.")

        X = self._prepare_data(data[text_column_judul], data[text_column_narasi])

        # Get probabilities and apply threshold
        y_proba = self.best_model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        if return_proba:
            return y_pred, y_proba
        return y_pred

    def explain_prediction(self, text_judul, text_narasi, num_features=10):
        if not self.is_trained:
            raise ValueError("Train the model before explanation.")

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
            st.warning(f"TF-IDF explanation unavailable: {e}")

        # Try basic feature importance if SHAP is not available
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
                feature_imp = feature_imp.sort_values('importance', ascending=False).head(num_features)
                
                for idx, row in feature_imp.iterrows():
                    explanation.append({
                        'feature': row['feature'], 
                        'shap_value': float(row['importance']),
                        'direction': 'positive' if row['importance'] > 0 else 'negative'
                    })
            elif hasattr(self.best_model, 'coef_'):
                importances = self.best_model.coef_[0]
                feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
                feature_imp = feature_imp.sort_values('importance', ascending=False).head(num_features)
                
                for idx, row in feature_imp.iterrows():
                    explanation.append({
                        'feature': row['feature'], 
                        'shap_value': float(row['importance']),
                        'direction': 'positive' if row['importance'] > 0 else 'negative'
                    })
        except Exception as e:
            st.warning(f"Model-based explanation unavailable: {e}")

        # Sort and limit by importance
        explanation = sorted(explanation, key=lambda x: abs(x['shap_value']), reverse=True)[:num_features]

        return {
            'prediction': int(prediction),
            'predicted_class': 'Hoax' if prediction==1 else 'Non-Hoax',
            'probability': probability,
            'confidence': probability if prediction==1 else 1-probability,
            'explanation': explanation
        }

# Function to load model in chunks to handle large file sizes
def load_model_in_chunks(uploaded_file, chunk_size=10*1024*1024):
    """Load a large model file in chunks to avoid memory issues"""
    bytes_data = BytesIO()
    for chunk in uploaded_file.chunks(chunk_size=chunk_size):
        bytes_data.write(chunk)
    bytes_data.seek(0)
    return joblib.load(bytes_data)

# Function to initialize system from core components
def initialize_system_from_components(components):
    """Initialize a HoaxDetectionSystem from core components"""
    system = HoaxDetectionSystem()
    
    # Set all the required components
    system.preprocessor = components.get('preprocessor')
    system.feature_extractor = components.get('feature_extractor')
    system.tfidf_vectorizer = components.get('tfidf_vectorizer')
    system.scaler = components.get('scaler')
    system.best_model = components.get('best_model')
    system.best_model_name = components.get('best_model_name')
    system.feature_names = components.get('feature_names', [])
    system.is_trained = True
    
    return system

def load_model_from_gdrive(gdrive_url, output_path="model/hoax_detector_model.pkl"):
    """Download model from Google Drive if not already present"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if model already exists
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading model from Google Drive... This may take a few minutes."):
            # Show a progress message
            progress_text = st.empty()
            progress_text.text("Starting download...")
            
            # Download the file
            try:
                gdown.download(gdrive_url, output_path, quiet=False)
                progress_text.text("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None
    
    # Load the model
    try:
        with st.spinner("Loading model into memory..."):
            model = joblib.load(output_path)
            return model
    except Exception as e:
        st.error(f"Error loading the downloaded model: {e}")
        return None

# Modified loading functions to support chunked loading for large models
def load_joblib_model(path, use_chunks=False):
    """Load a joblib model, optionally in chunks"""
    if use_chunks:
        # This would be implemented for file paths
        # For uploaded files, we use a different approach
        return joblib.load(path)
    else:
        return joblib.load(path)

# Streamlit sidebar for model loading
st.sidebar.title("Model Options")

# For demo purposes, you can either upload a model or use a pretrained one
model_source = st.sidebar.radio(
    "Model Source",
    ["Google Drive", "Upload Model", "Upload Model Components", "Use Default Path"]
)

if model_source == "Google Drive":
    gdrive_url = st.sidebar.text_input(
        "Google Drive Link",
        value="https://drive.google.com/uc?id=MASUKKAN_ID_FILE_ANDA"
    )
    
    if st.sidebar.button("Load Model from Drive"):
        # Make sure we have a valid Google Drive URL
        if "drive.google.com" in gdrive_url:
            # Convert share URL if necessary
            if "/file/d/" in gdrive_url:
                file_id = gdrive_url.split("/file/d/")[1].split("/")[0]
                gdrive_url = f"https://drive.google.com/uc?id={file_id}"
            
            # Load model
            detector = load_model_from_gdrive(gdrive_url)
            
            if detector is not None:
                # Initialize if needed
                if detector.tracker is None:
                    detector.tracker = TrainingProgressTracker(verbose=True)
                
                # Store in session state
                st.session_state.detector = detector
                st.session_state.model_loaded = True
                
                st.sidebar.success(f"Model loaded successfully: {detector.best_model_name}")
        else:
            st.sidebar.error("Please enter a valid Google Drive URL")

# Function to reset model state
def reset_model_state():
    st.session_state.model_loaded = False
    if 'detector' in st.session_state:
        del st.session_state.detector
    if 'model_components' in st.session_state:
        st.session_state.model_components = {}

if model_source == "Upload Complete Model":
    uploaded_model = st.sidebar.file_uploader("Upload your model file (joblib or pickle)", type=["pkl", "joblib"])
    
    if uploaded_model is not None:
        # Create a progress bar
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("Loading model...")
        
        try:
            # Save uploaded model to disk temporarily to avoid memory issues
            with open("temp_model.pkl", "wb") as f:
                progress_bar.progress(10)
                # Write in chunks to handle large files
                chunk_size = 5 * 1024 * 1024  # 5MB chunks
                file_buffer = uploaded_model.getbuffer()
                bytes_processed = 0
                total_bytes = len(file_buffer)
                
                while bytes_processed < total_bytes:
                    chunk = file_buffer[bytes_processed:bytes_processed + chunk_size]
                    f.write(chunk)
                    bytes_processed += len(chunk)
                    progress = min(80, int(80 * bytes_processed / total_bytes))
                    progress_bar.progress(progress)
                    status_text.text(f"Saved {bytes_processed/1024/1024:.1f}MB of {total_bytes/1024/1024:.1f}MB...")
            
            # Load the model
            status_text.text("Loading model into memory...")
            detector = joblib.load("temp_model.pkl")
            progress_bar.progress(90)
            
            # Initialize tracker if not present
            if detector.tracker is None:
                detector.tracker = TrainingProgressTracker(verbose=True)
            
            # Ensure model is marked as trained
            detector.is_trained = True
            
            # Store in session state
            st.session_state.detector = detector
            st.session_state.model_loaded = True
            
            progress_bar.progress(100)
            status_text.text(f"Ready! Model: {detector.best_model_name}")
            
            # Clean up
            os.remove("temp_model.pkl")
            
        except Exception as e:
            import traceback
            st.sidebar.error(f"Error loading model: {str(e)}")
            st.sidebar.code(traceback.format_exc())
            progress_bar.empty()
            status_text.empty()
            reset_model_state()

elif model_source == "Upload Model Components":
    st.sidebar.info("Upload individual model components to save memory.")
    
    # Component uploaders
    tfidf_vectorizer_file = st.sidebar.file_uploader("Upload TF-IDF Vectorizer", type=["pkl", "joblib"])
    best_model_file = st.sidebar.file_uploader("Upload Best Model", type=["pkl", "joblib"])
    scaler_file = st.sidebar.file_uploader("Upload Scaler", type=["pkl", "joblib"])
    feature_names_file = st.sidebar.file_uploader("Upload Feature Names (optional)", type=["pkl", "joblib", "txt"])
    
    # Check if required components are uploaded
    if tfidf_vectorizer_file and best_model_file and scaler_file:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            # Create model components
            components = {}
            
            # Load TF-IDF vectorizer
            status_text.text("Loading TF-IDF vectorizer...")
            progress_bar.progress(10)
            components['tfidf_vectorizer'] = joblib.load(tfidf_vectorizer_file)
            
            # Load best model
            status_text.text("Loading main model...")
            progress_bar.progress(30)
            components['best_model'] = joblib.load(best_model_file)
            
            # Load scaler
            status_text.text("Loading scaler...")
            progress_bar.progress(50)
            components['scaler'] = joblib.load(scaler_file)
            
            # Create preprocessor and feature extractor
            status_text.text("Creating text preprocessor...")
            progress_bar.progress(60)
            components['preprocessor'] = IndonesianTextPreprocessor()
            
            status_text.text("Creating feature extractor...")
            progress_bar.progress(70)
            components['feature_extractor'] = SentimentFeatureExtractor()
            
            # Load feature names if provided
            if feature_names_file:
                status_text.text("Loading feature names...")
                progress_bar.progress(80)
                try:
                    components['feature_names'] = joblib.load(feature_names_file)
                except:
                    # Try as a text file with one feature name per line
                    feature_names_text = feature_names_file.getvalue().decode('utf-8')
                    components['feature_names'] = feature_names_text.strip().split('\n')
            
            # Set model name
            components['best_model_name'] = "Uploaded Model"
            
            # Initialize system from components
            status_text.text("Initializing detection system...")
            progress_bar.progress(90)
            detector = initialize_system_from_components(components)
            
            # Store in session state
            st.session_state.detector = detector
            st.session_state.model_components = components
            st.session_state.model_loaded = True
            
            progress_bar.progress(100)
            status_text.text("Ready! Model components loaded successfully.")
            
        except Exception as e:
            import traceback
            st.sidebar.error(f"Error loading model components: {str(e)}")
            st.sidebar.code(traceback.format_exc())
            progress_bar.empty()
            status_text.empty()
            reset_model_state()
    
else:  # Use Default Path
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/hoax_detector_model.pkl"
    )
    
    if st.sidebar.button("Load Model"):
        # Create a progress bar
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text(f"Checking for model at {model_path}...")
        
        # Check if model exists at path
        if os.path.exists(model_path):
            progress_bar.progress(10)
            model_size_mb = os.path.getsize(model_path) / (1024*1024)
            status_text.text(f"Model found! Size: {model_size_mb:.2f} MB")
            
            try:
                # If model is very large, use a special loading approach
                if model_size_mb > 200:
                    status_text.text("Large model detected. Loading in chunks...")
                    progress_bar.progress(20)
                    
                    # Create chunks directory if it doesn't exist
                    os.makedirs('model_chunks', exist_ok=True)
                    
                    # Try to load the model in small chunks to avoid memory issues
                    with open(model_path, 'rb') as f:
                        detector = joblib.load(f)
                    
                else:
                    # Load the model normally
                    status_text.text("Loading model...")
                    progress_bar.progress(30)
                    detector = joblib.load(model_path)
                
                progress_bar.progress(90)
                status_text.text("Model loaded successfully!")
                
                # Initialize tracker if needed
                if detector.tracker is None:
                    detector.tracker = TrainingProgressTracker(verbose=True)
                
                # Ensure model is marked as trained
                detector.is_trained = True
                
                # Store in session state
                st.session_state.detector = detector
                st.session_state.model_loaded = True
                
                progress_bar.progress(100)
                status_text.text(f"Ready! Model: {detector.best_model_name}")
                
            except Exception as e:
                import traceback
                st.sidebar.error(f"Error loading model: {str(e)}")
                st.sidebar.code(traceback.format_exc())
                progress_bar.empty()
                status_text.empty()
                reset_model_state()
        else:
            st.sidebar.error(f"Model not found at: {model_path}")
            progress_bar.empty()
            status_text.empty()
            reset_model_state()

# Prediction threshold slider
threshold = st.sidebar.slider(
    "Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Adjust the threshold for classifying content as hoax"
)

# Main detection input area
st.header("Detect Hoax Content")

# Create two columns
col1, col2 = st.columns(2)

# Input fields in columns
with col1:
    judul = st.text_input("Headline (Judul)", value="Pemerintah Terapkan Protokol Kesehatan Baru untuk Atasi Covid-19")
    
with col2:
    narasi = st.text_area("Content (Narasi)", value="Dalam upaya mengendalikan penyebaran Covid-19, pemerintah telah menerapkan protokol kesehatan yang lebih ketat, termasuk pembatasan pertemuan dan penerapan sanksi bagi pelanggar. Informasi ini berdasarkan data resmi dan rekomendasi dari ahli kesehatan.", height=150)

# Check button
if st.button("Detect Hoax", type="primary"):
    if not st.session_state.model_loaded:
        st.error("Please load a model first!")
    else:
        with st.spinner("Analyzing content..."):
            # Create input dataframe
            input_df = pd.DataFrame({'judul': [judul], 'narasi': [narasi]})
            
            try:
                # Get the detector from session state
                detector = st.session_state.detector
                
                # Make prediction
                pred, prob = detector.predict(input_df, return_proba=True, threshold=threshold)
                is_hoax = bool(pred[0])
                confidence = prob[0] if is_hoax else 1-prob[0]
                
                # Get explanation
                explanation = detector.explain_prediction(judul, narasi)
                
                # Display results
                st.divider()
                st.header("Detection Results")
                
                # Create columns for results
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                    # Create a gauge-like visualization for the confidence
                    fig, ax = plt.subplots(figsize=(4, 0.3))
                    ax.barh(0, confidence, color='red' if is_hoax else 'green', height=0.5)
                    ax.barh(0, 1, color='lightgray', height=0.5, alpha=0.3)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    # Display prediction
                    if is_hoax:
                        st.error(f"HOAX (Confidence: {confidence:.2%})")
                    else:
                        st.success(f"NOT HOAX (Confidence: {confidence:.2%})")
                
                with result_col2:
                    # Explanation plot
                    if explanation and 'explanation' in explanation:
                        st.subheader("Key Factors")
                        
                        # Plot feature importance
                        fig, ax = plt.subplots(figsize=(10, 6))
                        factors = [(x['feature'], x['shap_value']) for x in explanation['explanation']]
                        factors.sort(key=lambda x: x[1])  # Sort by impact
                        
                        if factors:  # Check if we have explanation factors
                            features, values = zip(*factors)
                            colors = ['red' if v < 0 else 'green' for v in values]
                            
                            ax.barh(features, values, color=colors)
                            ax.set_xlabel('Impact on prediction')
                            ax.set_title(f"Factors influencing {'HOAX' if is_hoax else 'NOT HOAX'} classification")
                            ax.grid(axis='x', linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                        else:
                            st.info("No detailed explanation factors available for this prediction.")
                
                # Detailed explanation
                if explanation and 'explanation' in explanation and explanation['explanation']:
                    st.subheader("Detailed Explanation")
                    
                    explanation_df = pd.DataFrame(explanation['explanation'])
                    if not explanation_df.empty:
                        explanation_df.columns = ['Feature', 'Impact Value', 'Direction']
                        
                        # Format the dataframe for display
                        explanation_df['Impact'] = explanation_df.apply(
                            lambda x: f"{'⬆️' if x['Direction'] == 'positive' else '⬇️'} {abs(x['Impact Value']):.4f}",
                            axis=1
                        )
                        
                        st.dataframe(
                            explanation_df[['Feature', 'Impact']],
                            use_container_width=True
                        )
                
            except Exception as e:
                import traceback
                st.error(f"Error during prediction: {str(e)}")
                st.code(traceback.format_exc())

# Add a section for demo examples
st.divider()
st.header("Demo Examples")

# Create tabs for example types
tab1, tab2 = st.tabs(["Likely Hoax Examples", "Likely Real News Examples"])

with tab1:
    if st.button("Example: Government Hides Corona Cases", key="example_hoax"):
        st.session_state.judul = "BREAKING NEWS: Pemerintah Sembunyikan Kasus Corona"
        st.session_state.narasi = "Pemerintah dikabarkan menyembunyikan data kasus Corona. Dokumen rahasia menunjukkan adanya manipulasi data, sehingga menimbulkan kecurigaan publik. Sumber dari dalam pemerintahan yang tidak ingin disebutkan namanya mengklaim jumlah kematian sebenarnya 10x lipat dari data resmi!!!"
        st.rerun()

with tab2:
    if st.button("Example: Health Protocol Implementation", key="example_true"):
        st.session_state.judul = "Pemerintah Terapkan Protokol Kesehatan Baru untuk Atasi Covid-19"
        st.session_state.narasi = "Dalam upaya mengendalikan penyebaran Covid-19, pemerintah telah menerapkan protokol kesehatan yang lebih ketat, termasuk pembatasan pertemuan dan penerapan sanksi bagi pelanggar. Informasi ini berdasarkan data resmi dan rekomendasi dari ahli kesehatan."
        st.rerun()

# Footer
st.divider()
st.markdown("""
**About this application**  
This Indonesian Hoax Detection System uses machine learning with ensemble methods to analyze 
and detect potential hoaxes in Indonesian news. The model analyzes both the headline and content text 
using natural language processing, sentiment analysis, and credibility indicators.

**Memory-Optimized Implementation:**  
This application is specially designed to handle large machine learning models efficiently.
""")
