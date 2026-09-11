

import joblib
import re
import string
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# ---------------------------------------------------
# Model aur vectorizer ek baar load karo (baar baar load na ho)
# ---------------------------------------------------
MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ---------------------------------------------------
# Same cleaning function jo preprocessing.py mein tha
# (train aur predict time pe cleaning SAME honi chahiye)
# ---------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)


# ---------------------------------------------------
# Main prediction function
# ---------------------------------------------------
def predict_news(text):
    """
    Input: raw news text (title + content, ya sirf content)
    Output: dictionary with label aur confidence score
    """
    # Text clean karo
    cleaned = clean_text(text)

    # TF-IDF mein convert karo (SAME vectorizer jo training mein use hua tha)
    text_tfidf = vectorizer.transform([cleaned])

    # Prediction karo
    prediction = model.predict(text_tfidf)[0]

    # Confidence score nikalo (probability)
    probabilities = model.predict_proba(text_tfidf)[0]
    classes = model.classes_

    # Predicted class ki probability nikalo
    confidence = max(probabilities) * 100

    # Dono classes ki probability bhi bhejo (app mein dikhane ke liye)
    prob_dict = {classes[i]: round(probabilities[i] * 100, 2) for i in range(len(classes))}

    result = {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "probabilities": prob_dict
    }

    return result


# ---------------------------------------------------
# Direct testing ke liye
# ---------------------------------------------------
if __name__ == "__main__":
    sample_text = "Breaking news: Scientists discover shocking truth government hiding from you"

    result = predict_news(sample_text)

    print(f"Input Text: {sample_text}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Full Probabilities: {result['probabilities']}")
