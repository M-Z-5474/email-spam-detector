
# 📧 Email Spam Detector (Semi-Supervised Learning)

This project demonstrates how to detect **SPAM emails** using **semi-supervised machine learning**, even with limited labeled data. It combines advanced techniques like **Self-Training**, **SMOTE**, and **TF-IDF** vectorization for robust spam classification.

✅ **Live Demo**: [Streamlit App](https://email-spam-detector-9tmtkvxeaqjq8sq4gcvkqp.streamlit.app)

---

## 🎯 Objective

- Classify SMS/email messages into `spam` or `ham`
- Use only **20% labeled data** and **80% unlabeled data**
- Compare semi-supervised model vs. fully supervised + SMOTE
- Deploy a **real-time interactive app** to test email content

---

## 🚀 Project Highlights

| Feature | Description |
|--------|-------------|
| 🔍 Semi-Supervised Learning | Used `SelfTrainingClassifier` with smart labeling strategy |
| 🧪 Supervised Baselines | Logistic Regression and SMOTE-enhanced Logistic Regression |
| 🧠 TF-IDF Vectorization | Cleaned and vectorized all email texts |
| 📊 Model Evaluation | Precision, Recall, F1-score, Confusion Matrix |
| 🧱 Model Export | Saved final model (`clf_smote.pkl`) and `vectorizer.pkl` |
| 🌐 Deployed App | Built with Streamlit for real-time predictions |

---

## 🧠 Technologies Used

- Python (Scikit-learn, imbalanced-learn, NLTK)
- Semi-Supervised Learning: `SelfTrainingClassifier`
- Oversampling: `SMOTE`
- Text Vectorization: `TF-IDF`
- Web App: **Streamlit**
- Deployment: **Streamlit Cloud**

---

## 📂 Repository Structure

```bash
├── app.py                     # Streamlit app code
├── vectorizer.pkl             # TF-IDF vectorizer
├── clf_smote.pkl              # Final trained model (SMOTE + Logistic Regression)
├── requirements.txt           # Python dependencies
├── notebook/
│   └── spam_detection_notebook.ipynb  # Full training & evaluation notebook
│   └── README.md              # Notebook-specific documentation
├── README.md                  # You are here!
````

---

## 🧪 How to Use

### 🔹 1. Clone the Repo

```bash
git clone https://github.com/M-Z-5474/email-spam-detector.git
cd email-spam-detector
```

### 🔹 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 🔹 3. Run the App Locally

```bash
streamlit run app.py
```

---

## 📸 Streamlit App Preview

| Input Form                               | Prediction Result                          |
| ---------------------------------------- | ------------------------------------------ |
| ![form](https://i.imgur.com/Bg3qgHJ.png) | ![result](https://i.imgur.com/qzvlfMw.png) |

---

## 📓 Notebook for Training

All preprocessing, training logic, and model comparisons are included in:

📁 [`notebook/spam_detection_notebook.ipynb`](notebook/spam_detection_notebook.ipynb)

📄 [`notebook/README.md`](notebook/README.md)

---

## 🧑‍💻 Author

**Muhammad Zain Mushtaq**
📍 Researcher & AI/ML Enthusiast
🔗 [LinkedIn](https://www.linkedin.com/in/muhammad-zain-m-a75163358)
💼 [Portfolio](https://github.com/M-Z-5474)

---

## ⭐ Show Your Support

If you found this project helpful:

🌟 **Star** the repo
🔁 **Fork** it
🐞 Submit an issue or pull request

---

## 📬 License

This project is open-source and available under the [MIT License](LICENSE).

```

---

Would you like me to:

- Add a **badge section** (e.g., Streamlit, Python version)?
- Convert this into an actual `README.md` file you can copy-paste?
- Provide **images** (preview screenshots) for the Streamlit app?

Let me know how you’d like to finalize and polish it!
```
