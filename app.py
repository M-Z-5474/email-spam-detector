🟡 _Expected Result: SPAM_
""")

# Title
st.title("🚨 Spam Email Detector")
st.caption("Classify incoming emails as **SPAM** or **HAM** using a trained ML model.")

# Email input form
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

# Footer credit
st.markdown("---")
st.markdown(
    "Built with ❤️ by [Muhammad Zain Mushtaq](https://github.com/M-Z-5474) • "
    "[View GitHub Repo](https://github.com/M-Z-5474/email-spam-detector)"
)
