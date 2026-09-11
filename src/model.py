
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Humari pichli files se functions import kar rahe hain
from feature_extraction import feature_extraction_pipeline


# ---------------------------------------------------
# STEP 1: Model train karna
# ---------------------------------------------------
def train_model(X_train_tfidf, y_train):
    """
    Logistic Regression model train karta hai.
    Text classification ke liye ye simple aur effective model hai.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    print("Model training complete!")
    return model


# ---------------------------------------------------
# STEP 2: Model evaluate karna
# ---------------------------------------------------
def evaluate_model(model, X_test_tfidf, y_test):
    """
    Model ki performance check karta hai:
    - Accuracy
    - Precision, Recall, F1-score (har class ke liye)
    - Confusion Matrix
    """
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"MODEL ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*40}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy, y_pred


# ---------------------------------------------------
# STEP 3: Trained model ko save karna
# ---------------------------------------------------
def save_model(model, save_path):
    """
    Trained model ko .pkl file mein save karta hai.
    Isse Part 2 (app) mein load karke prediction karenge.
    """
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")


# ---------------------------------------------------
# Poora pipeline ek saath chalane ke liye
# ---------------------------------------------------
if __name__ == "__main__":
    PROCESSED_DATA = "data/processed/cleaned_data.csv"
    VECTORIZER_SAVE_PATH = "models/tfidf_vectorizer.pkl"
    MODEL_SAVE_PATH = "models/fake_news_model.pkl"

    # Feature extraction pipeline se data lo (Step 1 se 4 tak sab ho jayega)
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = feature_extraction_pipeline(
        PROCESSED_DATA, VECTORIZER_SAVE_PATH
    )

    # Model train karo
    model = train_model(X_train_tfidf, y_train)

    # Evaluate karo
    evaluate_model(model, X_test_tfidf, y_test)

    # Save karo
    save_model(model, MODEL_SAVE_PATH)

    print("\n✅ PART 1 COMPLETE! Model aur vectorizer dono 'models/' folder mein saved hain.")
    print("Ab Part 2 (app.py) shuru kar sakte ho.")
