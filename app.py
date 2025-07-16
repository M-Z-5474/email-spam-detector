import streamlit as st
import string
import joblib
import nltk
from nltk.corpus import stopwords

# Download stopwords (if not already present)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Load vectorizer and trained model
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("clf_smote.pkl")

# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words("english"))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Set page config
st.set_page_config(page_title="📧 Spam Email Detector", page_icon="📬")

# Sidebar navigation
option = st.sidebar.radio("📂 Navigation", ["🔍 Predict Email", "📌 Example Email", "ℹ️ About"])

# ------------------------- MAIN UI -------------------------

if option == "🔍 Predict Email":
    st.title("🚨 Spam Email Detector")
    st.caption("Classify incoming emails as **SPAM** or **HAM** using a trained ML model.")

    with st.form("email_form"):
        subject = st.text_input("✉️ Email Subject", placeholder="e.g., You’ve been selected to win a gift!")
        body = st.text_area("📄 Email Body", height=200, placeholder="e.g., Click here to claim your reward now!")
        submitted = st.form_submit_button("🔍 Analyze Email")

        if submitted:
            full_text = subject + " " + body
            clean = preprocess(full_text)
            vect = vectorizer.transform([clean])
            pred = model.predict(vect)[0]
            prob = model.predict_proba(vect)[0][pred] * 100

            st.markdown("## 🧪 Prediction Result")
            if pred == 1:
                st.error(f"🚨 This email is classified as **SPAM** with {prob:.2f}% confidence.")
            else:
                st.success(f"✅ This email is classified as **HAM (Not Spam)** with {prob:.2f}% confidence.")

elif option == "📌 Example Email":
    st.title("📌 Example Email")
    st.markdown("""
**Subject**: 🎉 Congratulations! You've won an iPhone!  
**Body**:  
You have been selected to receive a brand new iPhone 15 Pro Max!  
Claim your prize now at: http://fake-spam-link.com  
Offer valid for 24 hours only.

🟡 _Expected Result: SPAM_
""")

elif option == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
This project is a **Spam Email Classifier** built using a machine learning model trained on labeled email data.  
It uses **TF-IDF vectorization**, **SMOTE balancing**, and **Logistic Regression** for accurate classification.

---

### 👨‍💻 Developer Info

**Muhammad Zain Mushtaq**  
AI/ML & Data Science Enthusiast | IT Graduate  
📍 Pakistan  
🔗 [GitHub](https://github.com/M-Z-5474) | [LinkedIn](https://www.linkedin.com/in/muhammad-zain-m-a75163358/)
""")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ by [Muhammad Zain Mushtaq](https://github.com/M-Z-5474) • "
    "[View GitHub Repo](https://github.com/M-Z-5474/email-spam-detector)"
)
