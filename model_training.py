# DocuClassifyAI/model_training.py
import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# Download necessary NLTK data (run this once if you haven't)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# 1. Load Data (Subset of 20 Newsgroups)
categories = [
    'sci.med',
    'rec.sport.hockey',
    'talk.politics.misc',
    'comp.graphics'
]
print("Loading dataset...")
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42,
                                remove=('headers', 'footers', 'quotes'))
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42,
                               remove=('headers', 'footers', 'quotes'))

print(f"Training data size: {len(data_train.data)}")
print(f"Test data size: {len(data_test.data)}")
print(f"Categories: {data_train.target_names}")

# 2. Preprocessing and Feature Extraction (TF-IDF)
# We'll use scikit-learn's TfidfVectorizer which handles basic preprocessing.
# For more advanced preprocessing (like stemming), you'd create a custom tokenizer.
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

# 3. Model (Naive Bayes)
model = MultinomialNB()

# 4. Create a pipeline
# This combines the vectorizer and the model.
# It makes it easier to apply the same transformations to training and new data.
text_clf_pipeline = make_pipeline(vectorizer, model)

# 5. Train the model
print("Training model...")
text_clf_pipeline.fit(data_train.data, data_train.target)

# 6. Evaluate the model (optional, but good practice)
accuracy = text_clf_pipeline.score(data_test.data, data_test.target)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")

# 7. Save the trained model and vectorizer
output_dir = 'trained_model'
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, 'model.joblib')
vectorizer_path = os.path.join(output_dir, 'vectorizer.joblib') # Vectorizer is part of pipeline

joblib.dump(text_clf_pipeline, model_path)
print(f"Trained pipeline (vectorizer + model) saved to {model_path}")

# To demonstrate loading (not strictly needed in this script but good to know)
# loaded_pipeline = joblib.load(model_path)
# print("Model loaded successfully for a test prediction.")
# test_doc = ["The patient needs urgent surgery for his heart condition."]
# predicted_category_index = loaded_pipeline.predict(test_doc)[0]
# print(f"Test prediction: {data_train.target_names[predicted_category_index]}")
