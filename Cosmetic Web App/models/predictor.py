import os
import re
import joblib

# This dynamically finds the absolute path of the 'models' folder
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

def load_text_assets():
    stop_words = set()
    vocab_mapping = {}
    
    # 1. Load Stop Words
    stopwords_path = os.path.join(MODELS_DIR, 'stopwords_en.txt')
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stop_words = set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        print("⚠️ Warning: stopwords_en.txt not found.")

    # 2. Load Vocabulary Mapping
    vocab_path = os.path.join(MODELS_DIR, 'vocab.txt')
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if ':' in line:
                    parts = line.split(':')
                    word = parts[0].strip().lower()
                    try:
                        word_idx = int(parts[1].strip())
                    except ValueError:
                        word_idx = idx
                else:
                    word = line.lower()
                    word_idx = idx
                vocab_mapping[word] = word_idx
    except FileNotFoundError:
        print("⚠️ Warning: vocab.txt not found.")
        
    return stop_words, vocab_mapping

# Initialize and cache all model assets in memory once
try:
    title_tfidf = joblib.load(os.path.join(MODELS_DIR, 'title_tfidf.joblib'))
    glove_dict  = joblib.load(os.path.join(MODELS_DIR, 'glove_dict.joblib'))
    model_title = joblib.load(os.path.join(MODELS_DIR, 'model_title.joblib'))
    model_desc  = joblib.load(os.path.join(MODELS_DIR, 'model_desc.joblib'))
    model_num   = joblib.load(os.path.join(MODELS_DIR, 'model_num.joblib'))
    
    stop_words, vocab_dict = load_text_assets()
    print("✨ All machine learning weights successfully bound to application layer.")
except Exception as e:
    print(f"❌ Error loading assets from ./models/: {e}")

def clean_and_tokenize(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\-\s]', '', text)
    raw_words = text.split()
    
    # Filter out stopwords and words missing from the original vocab
    filtered_tokens = [
        word for word in raw_words 
        if word not in stop_words and word in vocab_dict
    ]
    return " ".join(filtered_tokens)

def predict_review(title_text):
    """
    Main pipeline entrypoint to classify a title.
    Returns 1 (BUYER) or 0 (NON-BUYER).
    """
    processed_title = clean_and_tokenize(title_text)
    title_vectorized = title_tfidf.transform([processed_title])
    prediction = model_title.predict(title_vectorized)[0]
    return int(prediction)