import streamlit as st
import string
import joblib
import nltk
from nltk.corpus import stopwords

# Download stopwords (once per session)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Load model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("clf_smote.pkl")

# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words("english"))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Streamlit page settings
st.set_page_config(page_title="📧 Spam Email Detector", page_icon="📬")

# Title and subtitle
st.title("🚨 Spam Email Detector")
st.caption("Classify emails as **SPAM** or **HAM** using a trained ML model.")
st.markdown("---")

# Email input form
st.subheader("📥 Analyze Your Email")
with st.form("email_form"):
    subject = st.text_input("✉️ Email Subject", placeholder="e.g., Congratulations! You've won a free gift!")
    body = st.text_area("📄 Email Body", height=180, placeholder="e.g., Click here to claim your iPhone before midnight!")
    submitted = st.form_submit_button("🔍 Analyze Email")

    if submitted:
        full_text = subject + " " + body
        clean = preprocess(full_text)
        vect = vectorizer.transform([clean])
        pred = model.predict(vect)[0]
        confidence = model.predict_proba(vect)[0][pred] * 100

        st.markdown("## 📊 Prediction Result")
        if pred == 1:
            st.error(f"🚨 This email is classified as **SPAM** with {confidence:.2f}% confidence.")
        else:
            st.success(f"✅ This email is classified as **HAM (Not Spam)** with {confidence:.2f}% confidence.")

# Example email guidance
st.markdown("---")
st.markdown("### 🧪 Example Input")
with st.expander("📌 Click to see an example"):
    st.markdown("""
**Subject**: 🎉 Congratulations! You've won an iPhone!  
**Body**:  
```text
You have been selected to receive a brand new iPhone 15 Pro Max!  
Claim your prize now at: http://fake-spam-link.com  
Offer valid for 24 hours only.
