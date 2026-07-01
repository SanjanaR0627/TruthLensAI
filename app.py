import pandas as pd
# ==========================================================
# TruthLens AI
# Part 1 - Imports, Configuration and Helper Functions
# ==========================================================

import streamlit as st
import pickle
import re
import string
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="📰",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    with open("truthlens_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model()

# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text

# ==========================================================
# SIMPLE SENTIMENT
# ==========================================================

def get_sentiment(text):

    positive_words = [
        "good","great","excellent","success",
        "positive","growth","benefit","win"
    ]

    negative_words = [
        "fake","fraud","crime","death",
        "attack","war","danger","loss"
    ]

    text = text.lower()

    positive = sum(word in text for word in positive_words)
    negative = sum(word in text for word in negative_words)

    if positive > negative:
        return "Positive"

    elif negative > positive:
        return "Negative"

    return "Neutral"

# ==========================================================
# TRUST SCORE
# ==========================================================

def calculate_trust_score(prediction, confidence):

    if prediction == "Real":
        score = confidence

    else:
        score = 100 - confidence

    return round(score,2)

# ==========================================================
# RISK LEVEL
# ==========================================================

def risk_level(score):

    if score >= 80:
        return "Low"

    elif score >= 60:
        return "Medium"

    else:
        return "High"

# ==========================================================
# TEXT STATISTICS
# ==========================================================

def text_statistics(text):

    words = len(text.split())

    characters = len(text)

    sentences = len(
        [s for s in text.split(".") if s.strip()]
    )

    return words, characters, sentences



def recommendation(prediction, trust):
    if prediction=="Real":
        return "Article appears credible. Verify with reliable sources."
    return "Article may be fake. Verify before sharing."

def explain(prediction, confidence, sentiment):
    return [f"Prediction: {prediction}",f"Confidence: {confidence:.2f}%",f"Sentiment: {sentiment}"]
# ==========================================================
# SAMPLE NEWS
# ==========================================================

sample_news = """
NASA successfully launched a new climate monitoring satellite today.

Scientists said the mission will improve weather forecasting
and help monitor climate change across the world.

The satellite is expected to remain operational for seven years.
"""
# ==========================================================
# TRUTHLENS AI - USER INTERFACE
# ==========================================================

st.title("📰 TruthLens AI")
st.markdown("### AI Powered Fake News Detection System")

st.markdown("---")

# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:

    st.header("📌 About")

    st.info(
        """
TruthLens AI uses a Machine Learning model to detect whether
a news article is Real or Fake.
"""
    )

    st.markdown("---")

    st.subheader("📊 Statistics")

    st.write("Articles Analyzed :", len(st.session_state.history))

    if st.button("🗑 Clear History"):

        st.session_state.history = []

        st.success("History Cleared")

# -------------------------------
# Sample News
# -------------------------------

sample_news = """
India successfully landed Chandrayaan-3 near the Moon's south pole.
The ISRO mission achieved all its planned scientific objectives and
received appreciation from scientists around the world.
"""

# -------------------------------
# Buttons
# -------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    analyze_btn = st.button(
        "🚀 Analyze News",
        use_container_width=True
    )

with col2:

    sample_btn = st.button(
        "📄 Load Sample",
        use_container_width=True
    )

with col3:

    clear_btn = st.button(
        "🗑 Clear News",
        use_container_width=True
    )

# -------------------------------
# Button Actions
# -------------------------------

if sample_btn:
    st.session_state.news_text = sample_news

if clear_btn:
    st.session_state.news_text = ""

# -------------------------------
# News Input
# -------------------------------

news = st.text_area(
    "Paste News Article",
    key="news_text",
    height=300,
    placeholder="Paste any news article here..."
)

st.markdown("---")
# ==========================================================
# ANALYZE NEWS
# ==========================================================

if analyze_btn:

    if news.strip() == "":

        st.warning("⚠ Please enter a news article.")

    else:

        with st.spinner("🤖 TruthLens AI is analyzing..."):

            # -----------------------------
            # Clean News
            # -----------------------------

            cleaned_news = clean_text(news)

            # -----------------------------
            # Vectorize
            # -----------------------------

            vector = vectorizer.transform([cleaned_news])

            # -----------------------------
            # Prediction
            # -----------------------------

            prediction_num = model.predict(vector)[0]

            # -----------------------------
            # Probability
            # -----------------------------

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(vector)[0]

                confidence = round(max(probability) * 100, 2)

                fake_probability = round(probability[0] * 100, 2)

                real_probability = round(probability[1] * 100, 2)

            else:

                confidence = 100

                fake_probability = 0

                real_probability = 100

            # -----------------------------
            # Label Mapping
            # -----------------------------

            if prediction_num == 1:

                prediction = "Real"

            else:

                prediction = "Fake"

            # -----------------------------
            # Sentiment
            # -----------------------------

            sentiment = get_sentiment(news)

            # -----------------------------
            # Trust Score
            # -----------------------------

            trust = calculate_trust_score(
                prediction,
                confidence
            )

            # -----------------------------
            # Risk
            # -----------------------------

            risk = risk_level(trust)

            # -----------------------------
            # Statistics
            # -----------------------------

            words, characters, sentences = text_statistics(news)

            # -----------------------------
            # Recommendation
            # -----------------------------

            advice = recommendation(
                prediction,
                trust
            )

            explanation = explain(
                prediction,
                confidence,
                sentiment
            )

            # -----------------------------
            # Save History
            # -----------------------------

            st.session_state.history.append({

                "Time": datetime.now().strftime("%d-%m-%Y %H:%M"),

                "Prediction": prediction,

                "Confidence": confidence

            })

            # ======================================================
            # RESULT
            # ======================================================

            st.markdown("---")

            if prediction == "Real":

                st.success("✅ REAL NEWS")

            else:

                st.error("❌ FAKE NEWS")

            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Confidence",
                    f"{confidence}%"
                )

            with col2:

                st.metric(
                    "Trust Score",
                    f"{trust}/100"
                )

            with col3:

                st.metric(
                    "Risk",
                    risk
                )

            st.progress(int(confidence))

            st.markdown("---")

            st.subheader("📊 Prediction Probability")

            st.write(f"🟢 Real : {real_probability}%")

            st.progress(real_probability/100)

            st.write(f"🔴 Fake : {fake_probability}%")

            st.progress(fake_probability/100)

            st.markdown("---")

            st.subheader("😊 Sentiment")

            st.info(sentiment)

            st.subheader("📄 Text Statistics")

            st.write(f"Words : {words}")

            st.write(f"Characters : {characters}")

            st.write(f"Sentences : {sentences}")

            st.subheader("💡 Recommendation")

            st.success(advice)

            st.subheader("🤖 AI Explanation")

            for item in explanation:

                st.write("✔", item)

            # -----------------------------
            # DEBUG
            # -----------------------------

            with st.expander("🔧 Debug Information"):

                st.write("Prediction Number :", prediction_num)

                st.write("Confidence :", confidence)

                if hasattr(model, "classes_"):

                    st.write("Classes :", model.classes_)

                if hasattr(model, "predict_proba"):

                    st.write("Probability :", probability)
# ==========================================================
# PREDICTION HISTORY
# ==========================================================

st.markdown("---")

st.subheader("📜 Prediction History")

if len(st.session_state.history) == 0:

    st.info("No articles analyzed yet.")

else:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

if len(st.session_state.history) > 0:

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download History",

        data=csv,

        file_name="TruthLens_Report.csv",

        mime="text/csv"

    )

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.markdown("---")

with st.expander("ℹ About TruthLens AI"):

    st.write("""

TruthLens AI is a Machine Learning based Fake News Detection System.

Features:

✅ Fake News Detection

✅ Confidence Score

✅ Trust Score

✅ Risk Analysis

✅ News Statistics

✅ Prediction History

✅ Download Report

Built using:

• Python

• Streamlit

• Scikit-Learn

• Pandas

• Machine Learning

""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

<h3>📰 TruthLens AI</h3>

AI Powered Fake News Detection System

Developed using ❤️ Streamlit & Machine Learning

</div>
""",
unsafe_allow_html=True
)