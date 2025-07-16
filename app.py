import streamlit as st
import string
import joblib
import nltk
from nltk.corpus import stopwords

# Load once to avoid repeated download
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Load saved model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("clf_smote.pkl")

# Preprocess function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Simulate extracting from a link (placeholder logic)
def extract_email_from_link(link):
    # In real usage, fetch from Gmail API or similar
    if "free-iphone" in link.lower():
        subject = "🎉Congratulations! You've WON a FREE iPhone!"
        body = "You’ve been selected as the lucky winner of an iPhone 15 Pro Max. Click here to claim!"
    else:
        subject = "Team Meeting Update"
        body = "Please find attached agenda for next week’s meeting. Regards, HR Team"
    return subject, body

# Streamlit App UI
st.set_page_config(page_title="📧 Spam Email Detector")
st.title("🚨 Spam Email Detector (with Confidence Score)")

tab1, tab2 = st.tabs(["📝 Enter Manually", "🔗 Estimate from Email Link"])

# ---- Tab 1: Manual Email Entry ----
with tab1:
    with st.form("manual_form"):
        subject = st.text_input("✉️ Subject")
        body = st.text_area("📄 Email Body", height=200)
        submitted = st.form_submit_button("🔍 Analyze")

        if submitted:
            full_text = subject + " " + body
            clean = preprocess(full_text)
            vect = vectorizer.transform([clean])
            pred = model.predict(vect)[0]
            proba = model.predict_proba(vect)[0]

            st.markdown("### 🧪 Prediction Result:")
            if pred == 1:
                st.error(f"🚨 Classified as **SPAM** ({proba[1]*100:.2f}% confidence)")
            else:
                st.success(f"✅ Classified as **HAM** ({proba[0]*100:.2f}% confidence)")

# ---- Tab 2: Estimate from Link ----
with tab2:
    with st.form("link_form"):
        link = st.text_input("🔗 Paste Email Link or Keyword")
        estimate = st.form_submit_button("🧪 Estimate")

        if estimate and link:
            subject, body = extract_email_from_link(link)
            full_text = subject + " " + body
            clean = preprocess(full_text)
            vect = vectorizer.transform([clean])
            pred = model.predict(vect)[0]
            proba = model.predict_proba(vect)[0]

            st.markdown("### 📬 Extracted Email:")
            st.write(f"**Subject:** {subject}")
            st.write(f"**Body:** {body}")

            st.markdown("### 🧪 Prediction Result:")
            if pred == 1:
                st.error(f"🚨 Classified as **SPAM** ({proba[1]*100:.2f}% confidence)")
            else:
                st.success(f"✅ Classified as **HAM** ({proba[0]*100:.2f}% confidence)")
