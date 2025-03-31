# Indonesian Hoax Detection System

# **https://detecthoax.streamlit.app/**

A Streamlit application for detecting hoaxes in Indonesian news using ensemble machine learning models and SHAP-based interpretations.

## Features

- Detect hoax vs. non-hoax content in Indonesian text
- Analyze both headlines and content body
- Provide detailed explanations of prediction factors
- Visualize feature importance with interactive charts
- Support for multiple model loading methods

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/indonesian-hoax-detection.git
   cd indonesian-hoax-detection
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Handling the Model

Due to the 400MB model size, you have multiple options for using the application:

1. **Host the model in Google Drive** (Recommended for Streamlit Cloud):
   - Upload your model file to Google Drive
   - Share the file with "Anyone with the link"
   - Note the file ID from the share link (e.g., in `https://drive.google.com/file/d/1a2b3c4d5e/view`, the ID is `1a2b3c4d5e`)
   - Use the "Google Drive Link" option in the app

2. **Local Model Path** (For local development):
   - Place your model file in a directory (e.g., `model/hoax_detector_model.pkl`)
   - Use the "Local Path" option in the app

3. **Direct Upload** (Not recommended for 400MB models):
   - Upload directly through the Streamlit interface
   - This may be slow or time out due to the file size

## Running the Application

### Local Development

Run the Streamlit app locally:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.


## Using the Application

1. Load your model using one of the available methods
2. Enter a headline and content text to analyze
3. Adjust the detection threshold if needed
4. Click "Detect Hoax" to get predictions
5. Explore the results and explanation visualizations

## Model Training

The model was trained on Indonesian news data using an ensemble approach with multiple classifiers:
- Random Forest
- XGBoost
- Logistic Regression
- SVM
- Gradient Boosting
- LightGBM

All combined using voting and stacking ensemble methods.

## Known Limitations

- The large model size (400MB) makes deployment challenging
- Initial loading time may be slow due to model size
- NLTK resources need to be downloaded on first run
- Shap interpretations can be computationally intensive
