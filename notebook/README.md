# 📓 Notebook: Semi-Supervised Email Spam Detection

This folder contains the complete development notebook used in **Email Spam Detection** — from data preprocessing and model training to evaluation and final model saving.

---

## 📘 Notebook Overview

### 🔍 Objective
To classify SMS/email messages as **SPAM** or **HAM**, using:
- **20% labeled data** and **80% unlabeled data** with semi-supervised learning
- Supervised models with and without SMOTE
- Final export of the best-performing model and vectorizer

---

## 🧱 Contents of This Notebook

| Section | Description |
|--------|-------------|
| ✅ Data Loading & Exploration | Reads and explores the SMS Spam Collection Dataset |
| 🧹 Preprocessing | Cleans and vectorizes text using TF-IDF |
| 🔁 Data Splitting | Performs stratified train/test split |
| 🧪 Semi-Supervised Learning | Trains `SelfTrainingClassifier` with smart label assignment |
| 🧪 Supervised Models | Trains standard Logistic Regression as a baseline |
| ⚖️ SMOTE Balancing | Uses SMOTE to address class imbalance and retrains Logistic Regression |
| 📊 Evaluation | Compares accuracy, precision, recall, F1-score, and confusion matrix |
| 💾 Model Export | Saves the final model (`clf_smote.pkl`) and vectorizer (`vectorizer.pkl`) for the Streamlit app |

---

## 💡 Why This Notebook Matters

- 🧠 Explains **how semi-supervised learning** can work with limited labels
- 📊 Shows **comparison with traditional supervised learning**
- 🔬 Uses **imbalanced learning techniques (SMOTE)** to boost minority class performance
- 💾 **Exports model artifacts** that are used in the deployed app

---

## 💾 Related Files
- `clf_smote.pkl`: Final trained model
- `vectorizer.pkl`: TF-IDF vectorizer
- `app.py`: Streamlit app using the above two files

---

## 🧑‍💻 Author
**Muhammad Zain Mushtaq**  
[LinkedIn](https://www.linkedin.com/in/muhammad-zain-m-a75163358/) • [GitHub](https://github.com/M-Z-5474)

---

> 🔁 This notebook complements the [main README](../README.md) and Streamlit app. Feel free to run, modify, and experiment!

