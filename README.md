# Mental Health Sentiment Analysis

This project implements an end-to-end mental health sentiment analysis system using machine learning and deep learning techniques. The system classifies text into mental-health-related categories and assigns a confidence score and risk level to each prediction. The project is designed for academic and research purposes and demonstrates the full pipeline from data preprocessing to model deployment.

Disclaimer: This project is intended strictly for educational and research use. It is not a diagnostic or clinical tool.

# Project Objectives

Analyze mental health–related text data

Classify text into predefined mental health categories

Provide confidence scores for predictions

Assign risk levels based on prediction confidence and category

Demonstrate real-time inference using a backend API and frontend interface

# Mental Health Categories

The model predicts the following classes:

depression

anxiety

stress

suicidal

normal

bipolar

ptsd

Each prediction includes a probability score and a derived risk level.

# Technology Stack

Python 3

PyTorch

Hugging Face Transformers (DistilBERT)

FastAPI

Uvicorn

HTML, CSS, JavaScript

Scikit-learn (baseline models)

# Project Folder Structure
MENTAL_HEALTH_SENTIMENT/
│
├── api/
│   ├── app.py
│   └── model/
│
├── data/
│   ├── raw/
│   │   ├── live_reddit_post.csv
│   │   ├── reddit_analysis.csv
│   │   └── sentiment_baseline.csv
│   │
│   └── processed/
│       ├── train_baseline.csv
│       └── val_baseline.csv
│
├── frontend/
│   └── index.html
│
├── models/
│   ├── final_model/
│   ├── final_model_domain_adapted/
│   ├── baseline_logreg.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── data_preprocessing_01.ipynb
│   ├── baseline_model_02.ipynb
│   ├── bert_training_03.ipynb
│   ├── model_comparison_04.ipynb
│   ├── bert_domain_adaption_05.ipynb
│   ├── model_comparison_06.ipynb
│   └── Final_evaluation_07.ipynb
│
├── src/
│
├── .gitignore
└── README.md

# Model Details

Baseline Model: TF-IDF + Logistic Regression

Deep Learning Model: DistilBERT

Task: Multi-class text classification

Fine-tuning: Domain-specific mental health text

Output: Predicted label with confidence score

# Risk Level Assignment

Risk levels are determined using rule-based logic combining prediction confidence and predicted class.

# Condition                                     # Risk Level
Suicidal content with high confidence	                 High
Mental health class with high confidence	             High
Mental health class with medium confidence	             Medium
Normal or low-confidence prediction	                     Low

This design prioritizes safety-aware interpretation.

# Backend API (FastAPI)
Start the server
cd api
uvicorn app:app --reload


The API runs at:

http://127.0.0.1:8000


API documentation is available at:

http://127.0.0.1:8000/docs

# Example API Request

POST /predict

{
  "text": "I feel overwhelmed and anxious most days"
}


Example response:

{
  "prediction": "anxiety",
  "confidence": 0.998,
  "risk_level": "High"
}

# Frontend

Start the FastAPI backend

Open the following file in a browser:

frontend/index.html


The frontend allows users to input text and view predictions, confidence scores, and risk levels.

# Model Files and Version Control

Trained model files are large and are excluded from version control using .gitignore.

To run the project locally:

Ensure model files are present inside the models/ directory

Update model paths in api/app.py if required

This follows standard machine learning best practices.

# Evaluation Summary

Baseline and deep learning models were evaluated

DistilBERT outperformed traditional models on contextual understanding

The system supports real-time inference

Designed with ethical and reproducibility considerations

# Ethical Considerations

No personally identifiable information is used

Data is treated as publicly available text

Clear disclaimer included

The system is not intended for clinical or diagnostic use

# Author

Honey Upadhyay
Post Graduate Diploma in Artificial Intelligence | B.Tech Graduate
Email: honeyupadhyay2002@gmail.com

# License

This project is intended for educational and research purposes only.
