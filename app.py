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

# Streamlit Page Config
st.set_page_config(page_title="📧 Spam Email Detector", page_icon="📬")

# Title and subtitle
st.title("🚨 Spam Email Detector")
st.markdown("**Detect SPAM or HAM from your emails using a trained machine learning model.**")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")

st.markdown("<br>", unsafe_allow_html=True)

# Input Form
st.markdown("## 📥 Input Email")
with st.form("email_form"):
    subject = st.text_input("✉️ Email Subject")
    body = st.text_area("📄 Email Body", height=200)
    submitted = st.form_submit_button("🔍 Analyze")

    if submitted:
        full_text = subject + " " + body
        clean = preprocess(full_text)
        vect = vectorizer.transform([clean])
        pred = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0][pred] * 100

        st.markdown("## 📊 Prediction Result")
        if pred == 1:
            st.error(f"🚨 This email is classified as **SPAM** with {proba:.2f}% confidence.")
        else:
            st.success(f"✅ This email is classified as **HAM (Not Spam)** with {proba:.2f}% confidence.")

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "Developed by [Muhammad Zain Mushtaq](https://github.com/M-Z-5474) • "
    "[GitHub Repository](https://github.com/M-Z-5474/email-spam-detector)"
)
