# 📚 NLP Projects Repository

This repository contains two separate machine learning projects focused on **Natural Language Processing (NLP)** and **Text Classification**:

1. **Spam - Ham Classification** — Classifying SMS messages as spam or not spam.  
2. **Kindle Reviews Sentiment Analysis** — Classifying Kindle product reviews into positive or negative sentiments.

---

## 🚩 Project 1: Spam vs Ham Classification

### 📌 Problem Statement
The objective of this project is to detect and classify SMS messages into **Spam** or **Ham (Not Spam)** categories. This helps in building filters that protect users from spam or phishing messages.

### 🔍 Project Workflow
1. Data Loading and Exploration
2. Data Cleaning & Preprocessing (stopwords removal, lemmatization)
3. Feature Extraction using **TF-IDF Vectorizer**
4. Model Training using ML algorithms
5. Evaluation using **Confusion Matrix, Precision, Recall, F1-score, and Accuracy**
6. Visualization of results

### 📁 Directory Structure
```
spam-ham-classification/
├── dataset/
│   └── spam.csv
├── spam_classifier.ipynb
└── README.md
```

### 📊 Dataset Details
- **Source**: Open-source dataset of SMS messages
- **Columns**: `label` (spam/ham), `message`
- **Format**: CSV

### 🔧 Technologies Used
- Python 3.x
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `sklearn`, `seaborn`, `matplotlib`

### 🚀 How to Run
```bash
cd spam-ham-classification
jupyter notebook spam_classifier.ipynb
```

### 📈 Model Performance
- Algorithms: **Naive Bayes**, **Logistic Regression**, **Random Forest**
- Achieved high accuracy in detecting spam messages.

---

## 🚩 Project 2: Kindle Reviews Sentiment Analysis

### 📌 Problem Statement
This project aims to classify **Kindle product reviews** into **Positive** or **Negative** sentiments. The goal is to understand customer satisfaction through sentiment analysis.

### 🔍 Project Workflow
1. Dataset Exploration
2. Data Cleaning (removing HTML tags, punctuations, stopwords)
3. Text Preprocessing (lowercasing, tokenization, lemmatization)
4. Feature Extraction using **TF-IDF**
5. Model Training using machine learning classifiers
6. Visualization of results with **WordClouds** and sentiment distribution
7. Performance evaluation with standard classification metrics

### 📁 Directory Structure
```
kindle-sentiment-analysis/
├── dataset/
│   └── kindle_reviews.csv
├── kindle_sentiment_analysis.ipynb
└── README.md
```

### 📊 Dataset Details
- **Source**: Open-source Kindle reviews dataset
- **Columns**: `review`, `sentiment`
- **Format**: CSV
- **Labeling**: Binary (Positive / Negative)

### 🔧 Technologies Used
- Python 3.x
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `sklearn`, `seaborn`, `matplotlib`, `wordcloud`

### 🚀 How to Run
```bash
cd kindle-sentiment-analysis
jupyter notebook kindle_sentiment_analysis.ipynb
```

### 📈 Model Performance
- Algorithms: **Logistic Regression**, **SVM**, **Random Forest**
- Visualizations: WordCloud for most common words in positive vs. negative reviews
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score


---

## 📊 Results Summary
| Project                        | Accuracy | Precision | Recall | F1-score |
|--------------------------------|----------|-----------|--------|----------|
| Spam vs Ham Classification     | ~80%     | High      | High   | High     |
| Kindle Sentiment Analysis      | ~80%     | High      | High   | High     |

---

## 🚧 Future Enhancements
✅ Implement deep learning models (LSTM, GRU, BERT)  
✅ Perform hyperparameter tuning for better results  
✅ Convert into deployable APIs or web apps  
✅ Incorporate larger datasets for robustness  

---

## 🛠️ Setup Instructions
1. Clone the repository:
```bash
git clone <your-repo-url>
```
2. Navigate to the desired project directory.
3. Open `.ipynb` files in Jupyter Notebook or Jupyter Lab.

---

## 💡 Key Learnings
- End-to-end text classification pipeline
- Feature engineering with TF-IDF
- Hands-on with classical ML models for NLP
- Visualizing data insights and results

---

## 🤝 Contributions
Feel free to fork the repository, suggest improvements, or submit pull requests.

---

## 📄 License
This repository is licensed under the **MIT License**.

---

## ✨ Acknowledgements
- Datasets sourced from public repositories.
- Inspired by common machine learning challenges in text classification.

---
