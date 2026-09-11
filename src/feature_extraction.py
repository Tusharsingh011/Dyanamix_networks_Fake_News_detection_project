
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ---------------------------------------------------
# STEP 1: Processed CSV load karna
# ---------------------------------------------------
def load_processed_data(file_path):
    """
    preprocessing.py se bani cleaned_data.csv ko load karta hai.
    """
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['clean_text', 'label'])  # safety check
    print(f"Processed data loaded. Shape: {df.shape}")
    return df


# ---------------------------------------------------
# STEP 2: Train-Test split
# ---------------------------------------------------
def split_data(df, test_size=0.2, random_state=42):
    """
    Data ko train aur test sets mein baantata hai.
    80% train, 20% test (default).
    """
    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y   # dono classes (real/fake) ka ratio train/test mein same rahega
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------
# STEP 3: TF-IDF Vectorizer banana aur fit karna
# ---------------------------------------------------
def create_tfidf_features(X_train, X_test, max_features=5000):
    """
    TF-IDF vectorizer ko sirf TRAIN data pe fit karta hai
    (test data pe sirf transform — taaki data leakage na ho).

    max_features: kitne top words rakhne hain (vocabulary size)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)   # unigrams + bigrams (single words + word pairs)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF feature matrix shape (train): {X_train_tfidf.shape}")
    print(f"TF-IDF feature matrix shape (test): {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, vectorizer


# ---------------------------------------------------
# STEP 4: Vectorizer ko save karna (Part 2 mein use hoga)
# ---------------------------------------------------
def save_vectorizer(vectorizer, save_path):
    """
    Fitted TF-IDF vectorizer ko save karta hai.
    Isse Part 2 (app) mein use karenge naya text transform karne ke liye.
    """
    joblib.dump(vectorizer, save_path)
    print(f"Vectorizer saved to: {save_path}")


# ---------------------------------------------------
# Poora pipeline ek saath chalane ke liye
# ---------------------------------------------------
def feature_extraction_pipeline(processed_data_path, vectorizer_save_path):
    df = load_processed_data(processed_data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(X_train, X_test)
    save_vectorizer(vectorizer, vectorizer_save_path)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# ---------------------------------------------------
# Directly run karne ke liye (test ke liye)
# ---------------------------------------------------
if __name__ == "__main__":
    PROCESSED_DATA = "data/processed/cleaned_data.csv"
    VECTORIZER_SAVE_PATH = "models/tfidf_vectorizer.pkl"

    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = feature_extraction_pipeline(
        PROCESSED_DATA, VECTORIZER_SAVE_PATH
    )

    print("\nFeature extraction complete! Ab model.py mein train karenge.")
