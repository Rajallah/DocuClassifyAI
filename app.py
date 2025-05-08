# DocuClassifyAI/app.py
import streamlit as st
import joblib
import os
import nltk # NLTK is still needed by scikit-learn's TfidfVectorizer if certain options are used,
            # or if we wanted to add custom tokenization later.
            # Ensure stopwords and punkt are downloaded as in model_training.py

# --- Page Configuration ---
st.set_page_config(
    page_title="DocuClassify AI",
    page_icon="📄",
    layout="centered"
)

# --- Global Variables & Model Loading ---
MODEL_DIR = 'trained_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')

# Define category names (must match the order from training)
# These come from `data_train.target_names` in model_training.py
CATEGORY_NAMES = [
    'comp.graphics',      # Mapped to 0 by scikit-learn (alphabetical order of chosen categories)
    'rec.sport.hockey',   # Mapped to 1
    'sci.med',            # Mapped to 2
    'talk.politics.misc'  # Mapped to 3
]
# If you are unsure of the order, print `data_train.target_names` in model_training.py
# after loading data `data_train = fetch_20newsgroups(...)`

@st.cache_resource # Cache the model loading for performance
def load_model_and_vectorizer():
    """Loads the pre-trained model pipeline."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please run model_training.py first.")
        return None
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        return model_pipeline
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model_pipeline = load_model_and_vectorizer()

# --- UI Elements ---
st.title("📄 DocuClassify AI")
st.subheader("By Yaseen A. Naser, Class 11.AI NV23112")
st.markdown("""
    Enter a piece of text (e.g., a news snippet, a paragraph) below,
    and the AI will try to classify it into one of the following categories:
    *   **Computer Graphics**
    *   **Hockey**
    *   **Medicine**
    *   **Politics**
""")

user_text = st.text_area("Enter your document text here:", height=200, key="doc_text_input")

if st.button("Classify Document", key="classify_button"):
    if model_pipeline is not None:
        if user_text.strip():
            # The input to predict must be a list or iterable of documents
            prediction_input = [user_text]
            
            try:
                with st.spinner("AI is thinking... 🤔"):
                    predicted_category_index = model_pipeline.predict(prediction_input)[0]
                    predicted_category_name = CATEGORY_NAMES[predicted_category_index]
                    
                    # Optional: Get probabilities for each class
                    # probabilities = model_pipeline.predict_proba(prediction_input)[0]
                    # prob_percent = probabilities[predicted_category_index] * 100

                st.success(f"**Predicted Category: {predicted_category_name}**")
                # st.info(f"Confidence: {prob_percent:.2f}%") # If you want to show confidence

                # st.subheader("How it works (Simplified):")
                # st.markdown("""
                # 1.  **Preprocessing:** Your text is cleaned (e.g., converted to lowercase).
                # 2.  **TF-IDF Vectorization:** The text is converted into numerical features that represent word importance.
                # 3.  **Naive Bayes Classifier:** A trained model predicts the most likely category based on these features.
                # """)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some text to classify.")
    else:
        st.error("Model is not loaded. Cannot classify.")

st.sidebar.header("About this Project")
st.sidebar.info("""
    This project demonstrates basic Natural Language Processing (NLP)
    and AI for document classification.
    - **Dataset:** 20 Newsgroups (subset)
    - **Technique:** TF-IDF + Naive Bayes
    - **Libraries:** Scikit-learn, NLTK, Streamlit
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Created by **Yaseen A. Naser**")
st.sidebar.markdown("Class 11.AI NV23112")

# --- To ensure NLTK resources are available for TfidfVectorizer ---
# This might be needed if TfidfVectorizer uses NLTK's tokenizer implicitly or if you extend it.
# It's good practice to ensure they are present.
def ensure_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        st.info("Downloading NLTK stopwords data...")
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        st.info("Downloading NLTK punkt tokenizer data...")
        nltk.download('punkt', quiet=True)

if model_pipeline: # Only run if model loaded, implies nltk might be needed
    ensure_nltk_data()
