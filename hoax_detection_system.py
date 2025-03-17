!pip install Sastrawi
!pip install streamlit
!pip install tensorflow
!pip install lightgbm
!pip install shap
!pip install ipywidgets

# Hoax Detection System for Indonesian Social Media Content
# Implementation based on sentiment analysis, ensemble models, and SHAP explanations

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import string
import time
import warnings
import joblib
from tqdm.notebook import tqdm
from collections import Counter
from typing import List, Dict, Tuple, Union, Any
import tensorflow as tf
from PIL import Image
from wordcloud import WordCloud

# For preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from nltk.probability import FreqDist
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# For feature extraction and modeling
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline

# For SHAP explanations
import shap

# For embedding visualization
from sklearn.manifold import TSNE

# Configure warnings and set display options
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
np.random.seed(42)
tf.random.set_seed(42)

# Check for GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Create stemmer and stopwords for Indonesian
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopwords_id = stopword_factory.get_stop_words()

# Add additional Indonesian stopwords
additional_stopwords = [
    'yg', 'dgn', 'nya', 'jd', 'klo', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
    'jg', 'utk', 'tdk', 'sdh', 'dr', 'pd', 'dlm', 'tsb', 'tp', 'kk', 'ju', 'sy',
    'jgn', 'ni', 'iy', 'bs', 'si', 'ya', 'lg', 'eh', 'kya', 'dah', 'loh', 'y', 'u'
]
stopwords_id.extend(additional_stopwords)

# Create Indonesian slang word dictionary
slang_words_dict = {
    'yg': 'yang', 'dgn': 'dengan', 'gak': 'tidak', 'ga': 'tidak', 'krn': 'karena',
    'udh': 'sudah', 'uda': 'sudah', 'udah': 'sudah', 'kalo': 'kalau',
    'klo': 'kalau', 'gtu': 'begitu', 'gitu': 'begitu', 'jd': 'jadi',
    'jgn': 'jangan', 'bs': 'bisa', 'utk': 'untuk', 'u': 'kamu',
    'km': 'kamu', 'kmu': 'kamu', 'sy': 'saya', 'ak': 'aku', 'aq': 'aku',
    'tp': 'tapi', 'tdk': 'tidak', 'pd': 'pada', 'dl': 'dulu', 'dlu': 'dulu',
    'org': 'orang', 'orng': 'orang', 'jg': 'juga', 'lg': 'lagi',
    'dgr': 'dengar', 'dr': 'dari', 'dlm': 'dalam', 'sm': 'sama', 'sdh': 'sudah',
    'sblm': 'sebelum', 'sih': 'sih', 'nih': 'ini', 'gt': 'begitu',
    'spt': 'seperti', 'skrg': 'sekarang', 'hrs': 'harus', 'msh': 'masih',
    'trs': 'terus', 'bnyk': 'banyak', 'byk': 'banyak', 'nmr': 'nomor',
    'blm': 'belum', 'bln': 'bulan', 'bbrp': 'beberapa', 'cm': 'cuma',
    'cma': 'cuma', 'emg': 'memang', 'pke': 'pakai', 'pake': 'pakai'
}

# Define emoticon and emoji dictionary 
emoticon_dict = {
    ':)': 'senang', ':-)': 'senang', ':D': 'senang', ':-D': 'sangat_senang',
    ':(': 'sedih', ':-(': 'sedih', ':\'(': 'menangis', ':"(': 'menangis',
    ':p': 'bercanda', ':-p': 'bercanda', ':o': 'kaget', ':O': 'kaget',
    ':3': 'imut', '<3': 'suka', ':/': 'bingung', ':\\': 'bingung',
    ';)': 'kedip', ';-)': 'kedip', '>:(': 'marah', '>:-(': 'marah',
    'xD': 'tertawa', 'XD': 'tertawa', '._.' : 'datar', '-_-': 'datar',
    '^_^': 'senang', 'o.O': 'bingung', 'O.o': 'bingung', ':*': 'cium'
}

# Create sentiment lexicon for Indonesian
# This is a simplified version; in practice, you'd use a more comprehensive lexicon
positive_words = [
    'baik', 'bagus', 'senang', 'gembira', 'indah', 'cantik', 'sukses', 'berhasil', 
    'setuju', 'benar', 'tepat', 'suka', 'cinta', 'sayang', 'peduli', 'terbaik',
    'kuat', 'ramah', 'bijaksana', 'adil', 'jujur', 'damai', 'sempurna', 'hebat',
    'istimewa', 'luar biasa', 'menyenangkan', 'mengagumkan', 'positif', 'aman'
]

negative_words = [
    'buruk', 'jelek', 'sedih', 'marah', 'benci', 'bodoh', 'gagal', 'salah', 
    'kecewa', 'susah', 'sulit', 'sakit', 'menderita', 'takut', 'cemas', 'khawatir',
    'lemah', 'jahat', 'kejam', 'tidak adil', 'bohong', 'berbahaya', 'kasar',
    'menyedihkan', 'menyebalkan', 'mengerikan', 'negatif', 'curiga', 'memalukan'
]

# Define clickbait and sensational words for Indonesian context
clickbait_words = [
    'wow', 'gila', 'mengejutkan', 'mencengangkan', 'viral', 'terbongkar',
    'rahasia', 'terkuak', 'terungkap', 'terheboh', 'terbaru', 'terpanas', 
    'menggemparkan', 'fantastis', 'spektakuler', 'tidak percaya', 'wajib', 
    'gawat', 'terkini', 'terpopuler', 'terlaris', 'terhebat', 'harus tahu',
    'bombastis', 'fenomenal', 'bikin kaget', 'inilah', 'begini', 'lihat',
    'dengar', 'tonton', 'shocking', 'tercengang', 'terkejut', 'wah'
]

hyperbolic_words = [
    'sangat', 'sekali', 'terlalu', 'banget', 'maha', 'super', 'ultra', 
    'mega', 'hiper', 'ekstrem', 'sempurna', 'total', 'mutlak', 'luar biasa', 
    'sangat2', 'terlampau', 'benar2', 'sungguh', 'teramat', 'amat'
]

# Define credibility indicators
credibility_negative = [
    'hoax', 'bohong', 'palsu', 'tipu', 'menipu', 'penipuan', 'penipu', 'dusta', 
    'fitnah', 'disinformasi', 'misinformasi', 'manipulasi', 'memanipulasi', 
    'sesat', 'menyesatkan', 'propaganda', 'rumor', 'gosip', 'isu', 'kabar burung',
    'tak terbukti', 'tidak terbukti', 'tak teruji', 'tidak teruji', 'konspirasi',
    'kontroversi', 'tidak benar', 'tak benar', 'mengelabui'
]

credibility_positive = [
    'fakta', 'bukti', 'terbukti', 'teruji', 'sumber', 'resmi', 'otentik', 'benar',
    'terpercaya', 'kredibel', 'sahih', 'valid', 'terverifikasi', 'verifikasi',
    'penelitian', 'peneliti', 'studi', 'survei', 'data', 'statistik', 'ilmiah', 
    'jurnal', 'akademis', 'akademik', 'ahli', 'pakar', 'terkonfirmasi'
]

# Class for Text Preprocessing
class IndonesianTextPreprocessor:
    def __init__(self, 
                 remove_url=True, 
                 remove_html=True, 
                 remove_punctuation=True,
                 normalize_slang=True,
                 remove_stopwords=True,
                 stemming=True):
        self.remove_url = remove_url
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.normalize_slang = normalize_slang
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
    
    def preprocess(self, text):
        """
        Apply preprocessing steps to the text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if self.remove_url:
            text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<.*?>', ' ', text)
        
        # Replace emoticons with their meaning
        for emoticon, meaning in emoticon_dict.items():
            text = text.replace(emoticon, ' ' + meaning + ' ')
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize slang words
        if self.normalize_slang:
            words = text.split()
            text = ' '.join([slang_words_dict.get(word, word) for word in words])
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in stopwords_id]
        
        # Stemming
        if self.stemming:
            tokens = [stemmer.stem(word) for word in tokens]
        
        # Join tokens back to text
        text = ' '.join(tokens)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def fit_transform(self, texts):
        """
        Apply preprocessing to a list of texts
        """
        return [self.preprocess(text) for text in texts]

 # Feature extractor class for sentiment and linguistic features
class SentimentFeatureExtractor:
    def __init__(self, 
                 extract_sentiment=True,
                 extract_linguistic=True,
                 extract_credibility=True):
        self.extract_sentiment = extract_sentiment
        self.extract_linguistic = extract_linguistic
        self.extract_credibility = extract_credibility
        self.feature_names = []
    
    def _count_words(self, text, word_list):
        """Count occurrences of words from a list in the text"""
        if not isinstance(text, str):
            return 0
        count = 0
        for word in word_list:
            count += sum(1 for _ in re.finditer(r'\b' + re.escape(word) + r'\b', text.lower()))
        return count
    
    def _extract_sentiment_features(self, text):
        """Extract sentiment-related features from text"""
        if not isinstance(text, str):
            text = ""
            
        features = {}
        
        # Positive and negative word counts
        features['positive_count'] = self._count_words(text, positive_words)
        features['negative_count'] = self._count_words(text, negative_words)
        
        # Calculate sentiment ratio if there are any sentiment words
        total_sentiment_words = features['positive_count'] + features['negative_count']
        if total_sentiment_words > 0:
            features['positive_ratio'] = features['positive_count'] / total_sentiment_words
            features['negative_ratio'] = features['negative_count'] / total_sentiment_words
        else:
            features['positive_ratio'] = 0
            features['negative_ratio'] = 0
            
        # Sentiment score (-1 to 1, where -1 is very negative, 1 is very positive)
        if total_sentiment_words > 0:
            features['sentiment_score'] = (features['positive_count'] - features['negative_count']) / total_sentiment_words
        else:
            features['sentiment_score'] = 0
            
        return features
    
    def _extract_linguistic_features(self, text):
        """Extract linguistic features from text"""
        if not isinstance(text, str):
            text = ""
            
        features = {}
        
        # Basic counts
        features['char_count'] = len(text)
        
        # Word count
        words = text.split()
        features['word_count'] = len(words)
        
        # Average word length
        if features['word_count'] > 0:
            features['avg_word_length'] = sum(len(word) for word in words) / features['word_count']
        else:
            features['avg_word_length'] = 0
            
        # Sentence count
        sentences = re.split(r'[.!?]+', text)
        features['sentence_count'] = sum(1 for s in sentences if s.strip())
        
        # Average sentence length
        if features['sentence_count'] > 0:
            features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        else:
            features['avg_sentence_length'] = 0
            
        # Punctuation counts
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        
        # Capitalization ratio
        if features['char_count'] > 0:
            features['uppercase_ratio'] = features['uppercase_count'] / features['char_count']
        else:
            features['uppercase_ratio'] = 0
            
        # Special pattern counts
        features['all_caps_words'] = sum(1 for word in words if word.isupper() and len(word) > 1)
        
        return features
    
    def _extract_credibility_features(self, text):
        """Extract credibility-related features from text"""
        if not isinstance(text, str):
            text = ""
            
        features = {}
        
        # Count clickbait and sensational words
        features['clickbait_count'] = self._count_words(text, clickbait_words)
        features['hyperbolic_count'] = self._count_words(text, hyperbolic_words)
        
        # Count credibility indicators
        features['credibility_negative'] = self._count_words(text, credibility_negative)
        features['credibility_positive'] = self._count_words(text, credibility_positive)
        
        # Credibility score
        total_cred_words = features['credibility_positive'] + features['credibility_negative']
        if total_cred_words > 0:
            features['credibility_score'] = (features['credibility_positive'] - features['credibility_negative']) / total_cred_words
        else:
            features['credibility_score'] = 0
        
        # Calculate ratios for clickbait and hyperbolic words
        word_count = len(text.split())
        if word_count > 0:
            features['clickbait_ratio'] = features['clickbait_count'] / word_count
            features['hyperbolic_ratio'] = features['hyperbolic_count'] / word_count
        else:
            features['clickbait_ratio'] = 0
            features['hyperbolic_ratio'] = 0
            
        return features
    
    def extract_features(self, texts):
        """Extract all features from a list of texts"""
        feature_dict = []
        
        for text in texts:
            text_features = {}
            
            if self.extract_sentiment:
                text_features.update(self._extract_sentiment_features(text))
                
            if self.extract_linguistic:
                text_features.update(self._extract_linguistic_features(text))
                
            if self.extract_credibility:
                text_features.update(self._extract_credibility_features(text))
                
            feature_dict.append(text_features)
            
        # Store feature names if not already stored
        if not self.feature_names and feature_dict:
            self.feature_names = list(feature_dict[0].keys())
            
        return pd.DataFrame(feature_dict)
    
    def fit_transform(self, texts):
        """Alias for extract_features to match scikit-learn interface"""
        return self.extract_features(texts)

 # Main HoaxDetectionSystem class
class HoaxDetectionSystem:
    def __init__(self,
                 use_gpu=True,
                 handle_imbalance=None,  # 'none', 'class_weight', 'smote', 'smote_tomek', 'adasyn', 'undersample'
                 tfidf_params=None,
                 preprocessor_params=None,
                 feature_extractor_params=None):
        
        # Setup GPU usage if available
        self.use_gpu = use_gpu
        if use_gpu:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                print(f"Using GPU: {physical_devices}")
                # Allow dynamic memory growth if needed
                try:
                    for device in physical_devices:
                        tf.config.experimental.set_memory_growth(device, True)
                except:
                    print("Failed to set memory growth, using full GPU memory")
        
        # Parameters initialization
        self.handle_imbalance = handle_imbalance
        
        self.tfidf_params = {
            'max_features': 10000,
            'min_df': 2,
            'max_df': 0.95,
            'ngram_range': (1, 2)
        }
        if tfidf_params:
            self.tfidf_params.update(tfidf_params)
            
        self.preprocessor_params = {
            'remove_url': True,
            'remove_html': True,
            'remove_punctuation': True,
            'normalize_slang': True,
            'remove_stopwords': True,
            'stemming': True
        }
        if preprocessor_params:
            self.preprocessor_params.update(preprocessor_params)
            
        self.feature_extractor_params = {
            'extract_sentiment': True,
            'extract_linguistic': True,
            'extract_credibility': True
        }
        if feature_extractor_params:
            self.feature_extractor_params.update(feature_extractor_params)
        
        # Initialize components
        self.preprocessor = IndonesianTextPreprocessor(**self.preprocessor_params)
        self.feature_extractor = SentimentFeatureExtractor(**self.feature_extractor_params)
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.scaler = MinMaxScaler()
        
        # Models will be initialized during training
        self.base_models = {}
        self.ensemble_models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Model metadata
        self.feature_names = []
        self.is_trained = False
        self.class_balance = None
    
    def _combine_features(self, judul_tfidf, narasi_tfidf, meta_features, is_training=False):
        """Combine TF-IDF features with meta features"""
        # Konversi matriks TF-IDF ke dataframe dengan nama kolom string yang eksplisit
        judul_cols = [f"judul_tfidf_{i}" for i in range(judul_tfidf.shape[1])]
        judul_tfidf_df = pd.DataFrame(judul_tfidf.toarray(), columns=judul_cols)
        
        narasi_cols = [f"narasi_tfidf_{i}" for i in range(narasi_tfidf.shape[1])]
        narasi_tfidf_df = pd.DataFrame(narasi_tfidf.toarray(), columns=narasi_cols)
        
        # Gabungkan semua fitur
        combined = pd.concat([judul_tfidf_df, narasi_tfidf_df, meta_features], axis=1)
        
        # Pastikan semua nama kolom bertipe string
        combined.columns = combined.columns.astype(str)
        
        # Jika prediksi (bukan pelatihan) dan kita memiliki nama fitur dari pelatihan
        if not is_training and hasattr(self, 'feature_names'):
            # Tambahkan kolom yang hilang dengan nilai 0
            for col in self.feature_names:
                if col not in combined.columns:
                    combined[col] = 0
            
            # Pilih hanya kolom yang ada saat pelatihan dengan urutan yang sama
            combined = combined[self.feature_names]
    
        return combined
    
    def _prepare_data(self, judul_series, narasi_series, labels=None):
        """Prepare data for model training or prediction"""
        # Tentukan apakah ini mode pelatihan atau prediksi
        is_training = labels is not None
        
        # Preprocess text
        preprocessed_judul = self.preprocessor.fit_transform(judul_series)
        preprocessed_narasi = self.preprocessor.fit_transform(narasi_series)
        
        # Ekstraksi fitur TF-IDF
        if is_training:
            # Saat pelatihan, fit vectorizer pada teks gabungan untuk kosakata konsisten
            all_texts = preprocessed_judul + preprocessed_narasi
            self.tfidf_vectorizer.fit(all_texts)
        
        # Transform teks menggunakan vectorizer
        judul_tfidf = self.tfidf_vectorizer.transform(preprocessed_judul)
        narasi_tfidf = self.tfidf_vectorizer.transform(preprocessed_narasi)
        
        # Ekstrak fitur meta (sentimen, linguistik, kredibilitas)
        judul_meta_features = self.feature_extractor.extract_features(judul_series)
        narasi_meta_features = self.feature_extractor.extract_features(narasi_series)
        combined_meta_features = pd.concat([judul_meta_features.add_prefix('judul_'), 
                                           narasi_meta_features.add_prefix('narasi_')], axis=1)
        
        # Scale fitur meta
        if is_training:
            scaled_features = self.scaler.fit_transform(combined_meta_features)
        else:
            scaled_features = self.scaler.transform(combined_meta_features)
        
        scaled_meta_features = pd.DataFrame(scaled_features, columns=combined_meta_features.columns)
        
        # Gabungkan semua fitur
        X = self._combine_features(judul_tfidf, narasi_tfidf, scaled_meta_features, is_training)
        
        if is_training:
            # Simpan nama fitur saat pelatihan
            self.feature_names = list(X.columns)
        
        if labels is not None:
            y = labels
            return X, y
        else:
            return X
        
    def train(self, data, target_column='label', text_column_judul='judul', text_column_narasi='narasi', 
              test_size=0.2, random_state=42):
        """Train the hoax detection models"""
        print("Starting training process...")
        start_time = time.time()
        
        # Extract features for judul and narasi
        X, y = self._prepare_data(data[text_column_judul], data[text_column_narasi], data[target_column])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Simpan sampel kecil dari data latih untuk SHAP (tambahkan baris ini)
        self.X_train_sample = X_train.sample(min(20, len(X_train)), random_state=random_state)
        
        # Analyze class distribution
        self.class_balance = dict(Counter(y_train))
        print(f"Class distribution in training set: {self.class_balance}")
        
        # Handle class imbalance if specified
        if self.handle_imbalance and self.handle_imbalance != 'none':
            X_train_resampled, y_train_resampled = self._handle_imbalance(X_train, y_train)
            print(f"After handling imbalance: {dict(Counter(y_train_resampled))}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Train base models
        print("Training base models...")
        self._train_base_models(X_train_resampled, y_train_resampled, X_test, y_test)
        
        # Train ensemble models
        print("Training ensemble models...")
        self._train_ensemble_models(X_train_resampled, y_train_resampled, X_test, y_test)
        
        # Select best model
        print("Evaluating and selecting best model...")
        self._select_best_model(X_test, y_test)
        
        self.is_trained = True
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def _handle_imbalance(self, X, y):
        """Apply the specified strategy to handle class imbalance"""
        if self.handle_imbalance == 'class_weight':
            # Class weight handled in model parameters
            return X, y
        
        elif self.handle_imbalance == 'smote':
            print("Applying SMOTE for oversampling...")
            smote = SMOTE(random_state=42)
            return smote.fit_resample(X, y)
        
        elif self.handle_imbalance == 'smote_tomek':
            print("Applying SMOTE-Tomek for hybrid sampling...")
            smote_tomek = SMOTETomek(random_state=42)
            return smote_tomek.fit_resample(X, y)
        
        elif self.handle_imbalance == 'adasyn':
            print("Applying ADASYN for adaptive oversampling...")
            adasyn = ADASYN(random_state=42)
            return adasyn.fit_resample(X, y)
        
        elif self.handle_imbalance == 'undersample':
            print("Applying random undersampling...")
            undersampler = RandomUnderSampler(random_state=42)
            return undersampler.fit_resample(X, y)
        
        else:
            print(f"Unknown imbalance handling strategy: {self.handle_imbalance}. Using original data.")
            return X, y
    
    def _train_base_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate individual base models"""
        # Define class weights if needed
        if self.handle_imbalance == 'class_weight':
            class_weight = {0: 1, 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}
            print(f"Using class weights: {class_weight}")
        else:
            class_weight = None
        
        # Define base models
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, class_weight=class_weight, random_state=42, n_jobs=-1
            ),
            'multinomial_nb': MultinomialNB(),
            'svm_linear': SVC(
                kernel='linear', probability=True, class_weight=class_weight, random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, class_weight=class_weight, random_state=42, n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'
            ),
            'lightgbm': lgbm.LGBMClassifier(
                n_estimators=100, class_weight=class_weight, random_state=42, n_jobs=-1
            ),
            'adaboost': AdaBoostClassifier(
                n_estimators=100, random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
            )
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_prob)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score
            }
            
            # Store model
            self.base_models[name] = model
            
            # Print results
            print(f"  {name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
        
        # Display results as a dataframe
        results_df = pd.DataFrame(results).T
        print("\nBase Models Performance:")
        print(results_df.sort_values('f1', ascending=False))
        
        return results_df
    
    def _train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate ensemble models"""
        if not self.base_models:
            raise ValueError("Base models must be trained before ensemble models")
        
        # Define ensemble models
        voting_classifiers = {
            'voting_hard': VotingClassifier(
                estimators=[
                    ('logistic_regression', self.base_models['logistic_regression']),
                    ('random_forest', self.base_models['random_forest']),
                    ('xgboost', self.base_models['xgboost'])
                ],
                voting='hard'
            ),
            'voting_soft': VotingClassifier(
                estimators=[
                    ('logistic_regression', self.base_models['logistic_regression']),
                    ('random_forest', self.base_models['random_forest']),
                    ('xgboost', self.base_models['xgboost'])
                ],
                voting='soft'
            )
        }
        
        # Stacking classifier
        stacking_classifier = StackingClassifier(
            estimators=[
                ('logistic_regression', self.base_models['logistic_regression']),
                ('random_forest', self.base_models['random_forest']),
                ('xgboost', self.base_models['xgboost']),
                ('lightgbm', self.base_models['lightgbm']),
                ('svm_linear', self.base_models['svm_linear'])
            ],
            final_estimator=LogisticRegression()
        )
        
        voting_classifiers['stacking'] = stacking_classifier
        
        # Train and evaluate ensemble models
        results = {}
        
        for name, model in voting_classifiers.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                # For hard voting which doesn't have predict_proba
                y_prob = None
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            if y_prob is not None:
                auc_score = roc_auc_score(y_test, y_prob)
            else:
                auc_score = None
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score
            }
            
            # Store model
            self.ensemble_models[name] = model
            
            # Print results
            if auc_score:
                print(f"  {name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                      f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
            else:
                print(f"  {name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                      f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Display results as a dataframe
        results_df = pd.DataFrame(results).T
        print("\nEnsemble Models Performance:")
        print(results_df.sort_values('f1', ascending=False))
        
        return results_df
    
    def _select_best_model(self, X_test, y_test):
        """Select the best model based on F1 score"""
        all_models = {**self.base_models, **self.ensemble_models}
        
        best_f1 = 0
        best_model = None
        best_name = None
        
        for name, model in all_models.items():
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = name
        
        self.best_model = best_model
        self.best_model_name = best_name
        
        print(f"Best model: {best_name} with F1 score: {best_f1:.4f}")
        
        return best_model, best_name
    
    def predict(self, data, text_column_judul='judul', text_column_narasi='narasi', return_proba=False):
        """Predict using the best model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare data
        X = self._prepare_data(data[text_column_judul], data[text_column_narasi])
        
        # Make predictions
        y_pred = self.best_model.predict(X)
        
        if return_proba:
            y_proba = self.best_model.predict_proba(X)[:, 1]
            return y_pred, y_proba
        else:
            return y_pred
    
    def evaluate(self, data, target_column='label', text_column_judul='judul', text_column_narasi='narasi', threshold=0.5):
        """Evaluate the model on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Prepare data
        X, y_true = self._prepare_data(data[text_column_judul], data[text_column_narasi], data[target_column])
        
        # All evaluation metrics
        evaluation = {}
        
        # For each model
        all_models = {**self.base_models, **self.ensemble_models, 'best_model': self.best_model}
        
        for name, model in all_models.items():
            print(f"Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            # Apply custom threshold if needed
            if threshold != 0.5:
                y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate metrics
            conf_matrix = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc_score = roc_auc_score(y_true, y_prob)
            
            # Store all metrics
            evaluation[name] = {
                'confusion_matrix': conf_matrix,
                'classification_report': report,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score
            }
            
            # Print results
            print(f"  Confusion matrix:\n{conf_matrix}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 score: {f1:.4f}")
            print(f"  AUC: {auc_score:.4f}")
        
        return evaluation
    
    def get_optimal_threshold(self, X, y_true):
        """Find the optimal classification threshold based on F1 score"""
        if not self.is_trained:
            raise ValueError("Model must be trained before finding optimal threshold")
        
        # Get probabilities
        y_prob = self.best_model.predict_proba(X)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        
        # Find best threshold
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_f1 = f1_scores[best_threshold_idx]
        
        print(f"Optimal threshold: {best_threshold:.2f} with F1 score: {best_f1:.4f}")
        
        return best_threshold
    
    def explain_prediction(self, text_judul, text_narasi, num_features=10):
        """
        Menjelaskan prediksi menggunakan pendekatan yang lebih robust
        """
        if not self.is_trained:
            raise ValueError("Model harus dilatih sebelum penjelasan")
        
        # Konversi input menjadi dataframe
        input_df = pd.DataFrame({'judul': [text_judul], 'narasi': [text_narasi]})
        
        # Preproses teks untuk analisis fitur langsung
        preprocessed_judul = self.preprocessor.preprocess(text_judul)
        preprocessed_narasi = self.preprocessor.preprocess(text_narasi)
        
        # Persiapkan data untuk prediksi
        X = self._prepare_data(input_df['judul'], input_df['narasi'])
        
        # Buat prediksi
        prediction = self.best_model.predict(X)[0]
        probability = self.best_model.predict_proba(X)[0, 1]
        
        # Inisialisasi list untuk penjelasan
        explanation = []
        
        # METODE 1: Analisis TF-IDF langsung untuk kata-kata penting
        # ----------------------------------------------------------
        # Dapatkan kata-kata dengan nilai TF-IDF tertinggi dari judul
        vectorizer = self.tfidf_vectorizer
        judul_tfidf = vectorizer.transform([preprocessed_judul])
        narasi_tfidf = vectorizer.transform([preprocessed_narasi])
        
        try:
            # Dapatkan nama fitur dari vectorizer
            feature_names = vectorizer.get_feature_names_out()
            
            # Untuk judul
            judul_important_indices = judul_tfidf.indices
            judul_important_values = judul_tfidf.data
            judul_important_words = [(feature_names[idx], judul_important_values[i]) 
                                    for i, idx in enumerate(judul_important_indices)]
            judul_important_words.sort(key=lambda x: x[1], reverse=True)
            
            # Untuk narasi
            narasi_important_indices = narasi_tfidf.indices
            narasi_important_values = narasi_tfidf.data
            narasi_important_words = [(feature_names[idx], narasi_important_values[i]) 
                                     for i, idx in enumerate(narasi_important_indices)]
            narasi_important_words.sort(key=lambda x: x[1], reverse=True)
            
            # Tambahkan kata-kata penting dari judul ke penjelasan
            for word, value in judul_important_words[:min(5, len(judul_important_words))]:
                # Sesuaikan nilai untuk mencerminkan prediksi
                adjusted_value = value * (1.0 if prediction == 1 else -1.0)
                explanation.append({
                    'feature': f'judul_kata_{word}',
                    'shap_value': float(adjusted_value),
                    'direction': 'positive' if adjusted_value > 0 else 'negative'
                })
            
            # Tambahkan kata-kata penting dari narasi ke penjelasan
            for word, value in narasi_important_words[:min(5, len(narasi_important_words))]:
                # Sesuaikan nilai untuk mencerminkan prediksi
                adjusted_value = value * (1.0 if prediction == 1 else -1.0)
                explanation.append({
                    'feature': f'narasi_kata_{word}',
                    'shap_value': float(adjusted_value),
                    'direction': 'positive' if adjusted_value > 0 else 'negative'
                })
        except Exception as e:
            print(f"Error saat menganalisis TF-IDF: {e}")
        
        # METODE 2: Ekstraksi fitur sentimen dan linguistik langsung
        # ----------------------------------------------------------
        # Ekstrak fitur sentimen dan linguistik
        judul_features = self.feature_extractor.extract_features([text_judul]).iloc[0]
        narasi_features = self.feature_extractor.extract_features([text_narasi]).iloc[0]
        
        # Fitur sentimen untuk judul
        if judul_features['sentiment_score'] != 0:
            explanation.append({
                'feature': 'judul_sentiment_score',
                'shap_value': float(judul_features['sentiment_score'] * (1.0 if prediction == 1 else -1.0)),
                'direction': 'positive' if (judul_features['sentiment_score'] > 0) == (prediction == 1) else 'negative'
            })
        
        # Fitur linguistik untuk judul
        if judul_features['uppercase_ratio'] > 0:
            explanation.append({
                'feature': 'judul_uppercase_ratio',
                'shap_value': float(judul_features['uppercase_ratio'] * 2.0 * (1.0 if prediction == 1 else -1.0)),
                'direction': 'positive' if prediction == 1 else 'negative'
            })
        
        if judul_features['exclamation_count'] > 0:
            explanation.append({
                'feature': 'judul_exclamation_count',
                'shap_value': float(judul_features['exclamation_count'] * 0.5 * (1.0 if prediction == 1 else -1.0)),
                'direction': 'positive' if prediction == 1 else 'negative'
            })
        
        # Fitur sentimen untuk narasi
        if narasi_features['sentiment_score'] != 0:
            explanation.append({
                'feature': 'narasi_sentiment_score',
                'shap_value': float(narasi_features['sentiment_score'] * (1.0 if prediction == 1 else -1.0)),
                'direction': 'positive' if (narasi_features['sentiment_score'] > 0) == (prediction == 1) else 'negative'
            })
        
        # Fitur kredibilitas untuk narasi
        if narasi_features['credibility_score'] != 0:
            explanation.append({
                'feature': 'narasi_credibility_score',
                'shap_value': float(narasi_features['credibility_score'] * -1.5 * (1.0 if prediction == 1 else -1.0)),
                'direction': 'negative' if prediction == 1 else 'positive'
            })
        
        # Fitur clickbait untuk narasi
        if narasi_features['clickbait_count'] > 0:
            explanation.append({
                'feature': 'narasi_clickbait_count',
                'shap_value': float(narasi_features['clickbait_count'] * 1.5 * (1.0 if prediction == 1 else -1.0)),
                'direction': 'positive' if prediction == 1 else 'negative'
            })
        
        # METODE 3: Pola penulisan hoaks yang terdeteksi
        # ----------------------------------------------
        # Deteksi pola penting seperti kapital berlebihan
        if narasi_features['uppercase_ratio'] > 0.1:
            explanation.append({
                'feature': 'narasi_uppercase_ratio',
                'shap_value': float(narasi_features['uppercase_ratio'] * 3.0),
                'direction': 'positive'
            })
        
        if narasi_features['all_caps_words'] > 2:
            explanation.append({
                'feature': 'narasi_all_caps_words',
                'shap_value': float(narasi_features['all_caps_words'] * 0.5),
                'direction': 'positive'
            })
        
        # Deteksi tanda seru berlebihan
        if narasi_features['exclamation_count'] > 2:
            explanation.append({
                'feature': 'narasi_exclamation_count',
                'shap_value': float(narasi_features['exclamation_count'] * 0.5),
                'direction': 'positive'
            })
        
        # Uji pola bahasa hoaks spesifik
        hoax_patterns = [
            "sebarkan", "viral", "rahasia", "terkuak", "terungkap", "dihapus", "dibatasi",
            "disensor", "dilarang", "tidak akan diberitakan", "media menutupi"
        ]
        
        for pattern in hoax_patterns:
            if pattern in text_narasi.lower():
                explanation.append({
                    'feature': f'narasi_pola_hoaks_{pattern}',
                    'shap_value': 2.0,
                    'direction': 'positive'
                })
        
        # Urutkan berdasarkan nilai absolut SHAP dan batasi jumlah
        explanation.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        explanation = explanation[:num_features]
        
        # Kembalikan hasil
        result = {
            'prediction': int(prediction),
            'predicted_class': 'Hoax' if prediction == 1 else 'Non-Hoax',
            'probability': probability,
            'confidence': probability if prediction == 1 else 1 - probability,
            'explanation': explanation
        }
        
        return result
    
    def save_model(self, filepath):
        """Save the entire model to a file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create model dict to save
        model_dict = {
            'preprocessor': self.preprocessor,
            'feature_extractor': self.feature_extractor,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler,
            'base_models': self.base_models,
            'ensemble_models': self.ensemble_models,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'class_balance': self.class_balance,
            'tfidf_params': self.tfidf_params,
            'preprocessor_params': self.preprocessor_params,
            'feature_extractor_params': self.feature_extractor_params,
            'handle_imbalance': self.handle_imbalance
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a model from a file"""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Create instance
        instance = cls(
            handle_imbalance=model_dict['handle_imbalance'],
            tfidf_params=model_dict['tfidf_params'],
            preprocessor_params=model_dict['preprocessor_params'],
            feature_extractor_params=model_dict['feature_extractor_params']
        )
        
        # Load model components
        instance.preprocessor = model_dict['preprocessor']
        instance.feature_extractor = model_dict['feature_extractor']
        instance.tfidf_vectorizer = model_dict['tfidf_vectorizer']
        instance.scaler = model_dict['scaler']
        instance.base_models = model_dict['base_models']
        instance.ensemble_models = model_dict['ensemble_models']
        instance.best_model = model_dict['best_model']
        instance.best_model_name = model_dict['best_model_name']
        instance.feature_names = model_dict['feature_names']
        instance.is_trained = model_dict['is_trained']
        instance.class_balance = model_dict['class_balance']
        
        print(f"Model loaded from {filepath}")
        return instance

 # Main execution
if __name__ == "__main__":
        # Set Pandas display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        
        # Load data
        print("Loading data...")
        train_data = pd.read_csv('Dataset/Data_latih.csv')
        test_data = pd.read_csv('Dataset/Data_uji.csv')
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Check if there are missing values
        print("\nMissing values in training data:")
        print(train_data.isnull().sum())
        
        # Fill missing values if any
        train_data['judul'].fillna('', inplace=True)
        train_data['narasi'].fillna('', inplace=True)
        
        # Check class distribution
        class_distribution = train_data['label'].value_counts(normalize=True)
        print("\nClass distribution:")
        print(class_distribution)
        
        # Based on the class distribution, decide whether to handle imbalance
        # If imbalance is severe (e.g., one class < 20%), use SMOTE or class weights
        major_class_pct = class_distribution.max()
        handle_imbalance = 'class_weight' if major_class_pct > 0.75 else None
        
        print(f"\nSelected imbalance handling strategy: {handle_imbalance}")
        
        # Initialize the hoax detection system
        hoax_detector = HoaxDetectionSystem(
            use_gpu=True,
            handle_imbalance=handle_imbalance,
            tfidf_params={
                'max_features': 10000,
                'min_df': 2,
                'max_df': 0.95,
                'ngram_range': (1, 2)
            }
        )
        
        # Train the model
        print("\nTraining the model...")
        hoax_detector.train(
            train_data, 
            target_column='label', 
            text_column_judul='judul', 
            text_column_narasi='narasi',
            test_size=0.2
        )
        
        # Save model
        hoax_detector.save_model('models/hoax_detector_model.pkl')
        
        # Make predictions on test set
        if 'label' in test_data.columns:
            # If test data has labels, evaluate the model
            print("\nEvaluating the model on test data...")
            eval_results = hoax_detector.evaluate(
                test_data,
                target_column='label',
                text_column_judul='judul',
                text_column_narasi='narasi'
            )
        else:
            # If test data doesn't have labels, just make predictions
            print("\nMaking predictions on test data...")
            predictions, probabilities = hoax_detector.predict(
                test_data,
                text_column_judul='judul',
                text_column_narasi='narasi',
                return_proba=True
            )
            
            # Add predictions to test data
            test_data['predicted_label'] = predictions
            test_data['probability'] = probabilities
            
            # Save predictions
            os.makedirs('predictions', exist_ok=True)
            test_data.to_csv('predictions/test_predictions.csv', index=False)
            print("Predictions saved to 'predictions/test_predictions.csv'")
        
        # Example of explaining a prediction
        example_judul = "BREAKING NEWS: Pemerintah Sembunyikan Kasus Corona"
        example_narasi = "Pemerintah sengaja memanipulasi data kasus Corona untuk menenangkan masyarakat. Ini terbukti dari dokumen rahasia yang bocor ke publik."
        
        print("\nExplaining a sample prediction:")
        explanation = hoax_detector.explain_prediction(example_judul, example_narasi)
        
        print(f"Prediction: {explanation['predicted_class']}")
        print(f"Confidence: {explanation['confidence']:.4f}")
        print("Top features influencing the prediction:")
        for feature in explanation['explanation']:
            direction = "+" if feature['direction'] == 'positive' else "-"
            print(f"  {direction} {feature['feature']}: {feature['shap_value']:.4f}")
