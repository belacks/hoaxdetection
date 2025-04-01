import os
import re
import string
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
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
""")

# Initialize session state for model loading status
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Download NLTK resources on app startup
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_resources()

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

nlp_resources = initialize_nlp_resources()

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

class HoaxDetectionSystem:
    def __init__(self, use_gpu=False, handle_imbalance=None, tfidf_params=None,
                 preprocessor_params=None, feature_extractor_params=None,
                 verbose=True, display_live_chart=False):
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.display_live_chart = display_live_chart
        self.handle_imbalance = handle_imbalance

        self.tfidf_params = {'max_features': 10000, 'min_df': 2, 'max_df': 0.95, 'ngram_range': (1, 2)}
        if tfidf_params:
            self.tfidf_params.update(tfidf_params)

        self.preprocessor_params = {'remove_url': True, 'remove_html': True, 'remove_punctuation': True,
                                    'normalize_slang': True, 'remove_stopwords': True, 'stemming': True}
        if preprocessor_params:
            self.preprocessor_params.update(preprocessor_params)

        self.feature_extractor_params = {'extract_sentiment': True, 'extract_linguistic': True,
                                         'extract_credibility': True}
        if feature_extractor_params:
            self.feature_extractor_params.update(feature_extractor_params)

        self.preprocessor = IndonesianTextPreprocessor(**self.preprocessor_params)
        self.feature_extractor = SentimentFeatureExtractor(**self.feature_extractor_params)
        
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.scaler = MinMaxScaler()

        self.base_models = {}
        self.ensemble_models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.is_trained = False
        self.class_balance = None
        self.tracker = TrainingProgressTracker(verbose=verbose)
        self.training_metrics = {
            'model_names': [],
            'accuracy': [],
            'f1_scores': [],
            'training_times': []
        }

    def _combine_features(self, judul_tfidf, narasi_tfidf, meta_features, is_training=False):
        judul_cols = [f"judul_tfidf_{i}" for i in range(judul_tfidf.shape[1])]
        judul_df = pd.DataFrame(judul_tfidf.toarray(), columns=judul_cols)
        narasi_cols = [f"narasi_tfidf_{i}" for i in range(narasi_tfidf.shape[1])]
        narasi_df = pd.DataFrame(narasi_tfidf.toarray(), columns=narasi_cols)
        combined = pd.concat([judul_df, narasi_df, meta_features], axis=1)
        combined.columns = combined.columns.astype(str)

        if not is_training and self.feature_names:
            for col in self.feature_names:
                if col not in combined.columns:
                    combined[col] = 0
            combined = combined[self.feature_names]

        return combined

    def _prepare_data(self, judul_series, narasi_series, labels=None):
        is_training = labels is not None
        
        preprocessed_judul = self.preprocessor.fit_transform(judul_series)
        preprocessed_narasi = self.preprocessor.fit_transform(narasi_series)

        judul_tfidf = self.tfidf_vectorizer.transform(preprocessed_judul)
        narasi_tfidf = self.tfidf_vectorizer.transform(preprocessed_narasi)

        judul_meta = self.feature_extractor.extract_features(judul_series).add_prefix('judul_')
        narasi_meta = self.feature_extractor.extract_features(narasi_series).add_prefix('narasi_')
        meta_features = pd.concat([judul_meta, narasi_meta], axis=1)

        scaled = self.scaler.transform(meta_features)
        scaled_meta = pd.DataFrame(scaled, columns=meta_features.columns)

        X = self._combine_features(judul_tfidf, narasi_tfidf, scaled_meta, is_training)

        if labels is not None:
            return X, labels
        return X

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
            st.error(f"Error in TF-IDF explanation: {e}")

        # Try SHAP explanation if possible
        try:
            explainer = shap.Explainer(self.best_model.predict_proba, X)
            shap_values = explainer(X)

            # Get top features by SHAP value
            shap_df = pd.DataFrame(shap_values.values[0], index=X.columns, columns=["shap_value"])
            shap_df["abs"] = shap_df["shap_value"].abs()
            top_features = shap_df.sort_values("abs", ascending=False).head(num_features)

            for idx, row in top_features.iterrows():
                explanation.append({'feature': idx, 'shap_value': row['shap_value'],
                                  'direction': 'positive' if row['shap_value']>0 else 'negative'})
        except Exception as e:
            st.error(f"Error in SHAP explanation: {e}")

        # Sort and limit by importance
        explanation = sorted(explanation, key=lambda x: abs(x['shap_value']), reverse=True)[:num_features]

        return {
            'prediction': int(prediction),
            'predicted_class': 'Hoax' if prediction==1 else 'Non-Hoax',
            'probability': probability,
            'confidence': probability if prediction==1 else 1-probability,
            'explanation': explanation
        }

# Streamlit sidebar for model loading
st.sidebar.title("Model Options")

# For demo purposes, you can either upload a model or use a pretrained one
model_source = st.sidebar.radio(
    "Model Source",
    ["Upload Model", "Use Default Model Path"]
)

if model_source == "Upload Model":
    uploaded_model = st.sidebar.file_uploader("Upload your model file (joblib or pickle)", type=["pkl", "joblib"])
    
    if uploaded_model is not None:
        # Create a progress bar
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("Loading model...")
        
        # Save uploaded model to disk temporarily
        with open("temp_model.pkl", "wb") as f:
            f.write(uploaded_model.getbuffer())
        
        progress_bar.progress(30)
        status_text.text("Model saved, initializing...")
        
        try:
            # Try to load the model
            detector = joblib.load("temp_model.pkl")
            progress_bar.progress(90)
            status_text.text("Model loaded successfully!")
            
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
            
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            progress_bar.empty()
            status_text.empty()
            
else:  # Use Default Model Path
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
            status_text.text(f"Model found! Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
            progress_bar.progress(30)
            status_text.text("Loading model...")
            
            try:
                # Load the model
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
                st.sidebar.error(f"Error loading model: {str(e)}")
                progress_bar.empty()
                status_text.empty()
        else:
            st.sidebar.error(f"Model not found at: {model_path}")
            progress_bar.empty()
            status_text.empty()

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
                        
                        features, values = zip(*factors)
                        colors = ['red' if v < 0 else 'green' for v in values]
                        
                        ax.barh(features, values, color=colors)
                        ax.set_xlabel('Impact on prediction')
                        ax.set_title(f"Factors influencing {'HOAX' if is_hoax else 'NOT HOAX'} classification")
                        ax.grid(axis='x', linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                
                # Detailed explanation
                st.subheader("Detailed Explanation")
                
                explanation_df = pd.DataFrame(explanation['explanation'])
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
        # Set values in the input fields
        st.session_state.judul_example = "BREAKING NEWS: Pemerintah Sembunyikan Kasus Corona"
        st.session_state.narasi_example = "Pemerintah dikabarkan menyembunyikan data kasus Corona. Dokumen rahasia menunjukkan adanya manipulasi data, sehingga menimbulkan kecurigaan publik. Sumber dari dalam pemerintahan yang tidak ingin disebutkan namanya mengklaim jumlah kematian sebenarnya 10x lipat dari data resmi!!!"
        
        # Rerun to update
        st.experimental_rerun()

with tab2:
    if st.button("Example: Health Protocol Implementation", key="example_true"):
        # Set values in the input fields
        st.session_state.judul_example = "Pemerintah Terapkan Protokol Kesehatan Baru untuk Atasi Covid-19"
        st.session_state.narasi_example = "Dalam upaya mengendalikan penyebaran Covid-19, pemerintah telah menerapkan protokol kesehatan yang lebih ketat, termasuk pembatasan pertemuan dan penerapan sanksi bagi pelanggar. Informasi ini berdasarkan data resmi dan rekomendasi dari ahli kesehatan."
        
        # Rerun to update
        st.experimental_rerun()

# Apply example values if they exist
if 'judul_example' in st.session_state:
    judul = st.session_state.judul_example
    st.text_input("Headline (Judul)", value=judul, key="judul_input")
    # Clear the session state
    del st.session_state.judul_example

if 'narasi_example' in st.session_state:
    narasi = st.session_state.narasi_example
    st.text_area("Content (Narasi)", value=narasi, key="narasi_input", height=150)
    # Clear the session state
    del st.session_state.narasi_example

# Footer
st.divider()
st.markdown("""
**About this application**  
This Indonesian Hoax Detection System uses machine learning with ensemble methods and SHAP interpretability 
to analyze and detect potential hoaxes in Indonesian news. The model analyzes both the headline and content text 
using natural language processing features, sentiment analysis, and credibility indicators.
""")
