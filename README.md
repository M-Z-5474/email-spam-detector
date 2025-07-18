
# 📧 Email Spam Detector (Semi-Supervised Learning)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/M-Z-5474/email-spam-detector)
![Repo Size](https://img.shields.io/github/repo-size/M-Z-5474/email-spam-detector)



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
**✅ Final Model Comparison**

| Model                          | Accuracy | Spam Recall | Spam F1  | Comment                               |
| ------------------------------ | -------- | ----------- | -------- | ------------------------------------- |
| Self-Training Classifier       | 95%      | 0.66        | 0.79     | Strong for limited labels             |
| Supervised LogisticRegression  | 95%      | 0.62        | 0.77     | Good baseline                         |
| **SMOTE + LogisticRegression** | **98%**  | **0.87**    | **0.91** | ⭐ Best performance (balanced & clean) |


🔥 SMOTE + Logistic Regression gives the best performance, especially for the minority class (spam), by fixing the imbalance problem.

---

## 📂 Repository Structure

```bash
├── app.py                     # Streamlit app code
├── vectorizer.pkl             # TF-IDF vectorizer
├── clf_smote.pkl              # Final trained model (SMOTE + Logistic Regression)
├── requirements.txt           # Python dependencies
├── notebook/
│   └── Email_Spam_Detection.ipynb  # Full training & evaluation notebook
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

| Input Form | Prediction Result |
|------------|------------------|
| ![form](https://raw.githubusercontent.com/M-Z-5474/email-spam-detector/main/assets/input_form.png) | ![result](https://raw.githubusercontent.com/M-Z-5474/email-spam-detector/main/assets/prediction_result.png) |

---

## 📓 Notebook for Training

All preprocessing, training logic, and model comparisons are included in:

📁 [`notebook/spam_detection_notebook.ipynb`](notebook/spam_detection_notebook.ipynb)

📄 [`notebook/README.md`](notebook/README.md)

---

## 🧑‍💻 Author

**Muhammad Zain Mushtaq**

📍 AI/ML & Data Scientist Enthusiast | Researcher 
🔗 [LinkedIn](https://www.linkedin.com/in/muhammad-zain-m-a75163358)
💼 [Portfolio](https://github.com/M-Z-5474)

---

## ⭐ Show Your Support

If you found this project helpful:

🌟 **Star** the repo
🔁 **Fork** it
🐞 Submit an issue or pull request

---


