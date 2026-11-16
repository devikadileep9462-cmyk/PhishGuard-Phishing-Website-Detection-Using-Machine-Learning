# 🛡️ PhishGuard – Phishing Website Detection Using Machine Learning

## 📌 Project Overview

PhishGuard is a machine learning-based web application designed to identify potentially malicious phishing websites by analyzing URL patterns.

The project uses URL-based feature extraction and machine learning algorithms to classify websites as **Phishing** or **Legitimate**. It provides a simple and user-friendly interface built with Streamlit, allowing users to enter a website URL and check its potential risk.

## 🎯 Objectives

- Detect potentially phishing websites using machine learning.
- Extract useful features from website URLs.
- Compare Random Forest and XGBoost classification models.
- Provide an easy-to-use interface for website risk checking.

## 🚀 Features

- Simple URL input interface.
- Automatic URL feature extraction.
- Phishing and legitimate website classification.
- Risk probability score.
- User-friendly Streamlit application.

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn

## 🤖 Machine Learning Models

The project compares two classification algorithms:

1. Random Forest Classifier
2. XGBoost Classifier

The best-performing model is selected based on cross-validation performance.

## ⚙️ Project Workflow

1. Dataset collection and exploratory data analysis.
2. Data preprocessing and cleaning.
3. URL-based feature extraction.
4. Model training and evaluation.
5. Model comparison and selection.
6. Streamlit application development.

## 📂 Project Structure

```text
PhishGuard/
│
├── app.py
├── url_features.py
├── train_url_model.py
├── eda.py
├── train.py
├── requirements.txt
├── README.md
└── screenshots/
```

## 💻 Installation and Setup

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project folder**

```bash
cd PhishGuard
```

**3. Install the required libraries**

```bash
pip install -r requirements.txt
```

**4. Train the URL-based model**

Make sure the dataset is available in the project folder.

```bash
python train_url_model.py
```

This generates the trained model file.

**5. Run the Streamlit application**

```bash
streamlit run app.py
```

## ⚠️ Limitations

PhishGuard uses URL-based patterns to estimate phishing risk. It does not inspect website content, page redirects, SSL certificates, or domain registration information.

The prediction is an estimate and should not be treated as a guarantee that a website is safe or malicious. Avoid entering sensitive information on suspicious websites.

## 🔮 Future Improvements

- Integrate real-time threat intelligence.
- Analyze website content and redirects.
- Improve URL feature extraction.
- Explore deep learning-based detection methods.
- Enhance model performance using larger datasets.

## 👩‍💻 Author

Developed as an individual machine learning project to explore practical applications of AI and cybersecurity.
