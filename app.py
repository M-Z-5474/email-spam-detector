import streamlit as st
import string
import joblib
import nltk
from nltk.corpus import stopwords

# Download stopwords (only if not already)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Load vectorizer and model
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("clf_smote.pkl")

# Text preprocessing
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Streamlit App
st.set_page_config(page_title="📧 Spam Email Detector")
st.title("🚨 Spam Email Detector (Semi-Supervised Model)")

with st.form("email_form"):
    subject = st.text_input("✉️ Email Subject")
    body = st.text_area("📄 Email Body", height=200)
    submitted = st.form_submit_button("🔍 Analyze")

    if submitted:
        full_text = subject + " " + body
        clean = preprocess(full_text)
        vect = vectorizer.transform([clean])
        pred = model.predict(vect)[0]

        st.markdown("### 🧪 Prediction:")
        if pred == 1:
            st.error("🚨 This email is classified as **SPAM**!")
        else:
            st.success("✅ This email is classified as **HAM (Not Spam)**.")
