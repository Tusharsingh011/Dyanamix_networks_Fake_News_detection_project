

import sys
import os

# src folder ko path mein add karo taaki predict.py import ho sake
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


import streamlit as st
from predict import predict_news, clean_text, model, vectorizer
from lime.lime_text import LimeTextExplainer


# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# ---------------------------------------------------
# Explainability function (LIME)
# ---------------------------------------------------
def explain_prediction(text):
    """
    LIME use karke batata hai ki model ne kaun se words dekh kar
    decision liya (real ya fake).
    """
    class_names = list(model.classes_)
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba_wrapper(texts):
        cleaned_texts = [clean_text(t) for t in texts]
        vectors = vectorizer.transform(cleaned_texts)
        return model.predict_proba(vectors)

    exp = explainer.explain_instance(
        text,
        predict_proba_wrapper,
        num_features=8
    )
    return exp

# ---------------------------------------------------
# UI - Header
# ---------------------------------------------------
st.title("📰 Fake News Detection System")
st.markdown("Enter a news article, headline, or claim below to check if it's likely **Real** or **Fake**.")
st.divider()


# ---------------------------------------------------
# UI - Text Input
# ---------------------------------------------------
user_input = st.text_area(
    "Enter News Text:",
    height=200,
    placeholder="Paste the news article, headline, or claim here..."
)

col1, col2 = st.columns([1, 1])
with col1:
    check_button = st.button("🔍 Check News", use_container_width=True)
with col2:
    explain_checkbox = st.checkbox("Show Explainability Insights")


# ---------------------------------------------------
# UI - Prediction Result
# ---------------------------------------------------
if check_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_news(user_input)

        st.divider()

        # Result display
        if result["prediction"] == "fake":
            st.error(f"### 🚨 Prediction: FAKE NEWS")
        else:
            st.success(f"### ✅ Prediction: REAL NEWS")

        # Confidence score
        st.metric(label="Confidence Score", value=f"{result['confidence']}%")

        # Progress bar for confidence
        st.progress(result['confidence'] / 100)

        # Full probability breakdown
        st.subheader("Probability Breakdown")
        for label, prob in result["probabilities"].items():
            st.write(f"**{label.capitalize()}**: {prob}%")
            st.progress(prob / 100)

        # Explainability section
        if explain_checkbox:
            st.divider()
            st.subheader("🔎 Explainability & Insights")
            st.caption("These are the words that most influenced the model's decision:")

            with st.spinner("Generating explanation..."):
                exp = explain_prediction(user_input)
                html_exp = exp.as_html()

            st.components.v1.html(html_exp, height=400, scrolling=True)


# ---------------------------------------------------
# Sidebar Info
# ---------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write("""
    This tool uses a Machine Learning model (Logistic Regression + TF-IDF)
    trained on a news dataset to classify text as **Real** or **Fake**.
    """)
    st.write("Built for College Final Year Project.")
