## Overview
This project aims to detect cyberbullying across various social media platforms using a multi-algorithmic machine learning approach. By leveraging different algorithms and pipeline methods, the project improves detection accuracy, creating a robust system to identify and mitigate harmful online behavior.

## Key Features
- Multi-algorithm approach for enhanced accuracy
- Data preprocessing pipeline for cleaning and transforming text
- Sentiment analysis for context-based detection
- Classification using various machine learning models
- Visualization of results for performance comparison

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning Algorithms:** Naive Bayes, Support Vector Machine (SVM), Random Forest, XGBoost
- **Text Processing:** Natural Language Processing (NLP), Tokenization, Lemmatization, Stop Word Removal
- **Deployment:** Jupyter Notebook, Google Colab

## Dataset
The dataset used for training and testing was sourced from publicly available social media text data. It includes labeled data for cyberbullying and non-cyberbullying content. 

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cyber-bully-detection.git


├── data/                   # Contains dataset files
├── models/                 # Contains saved machine learning models
├── notebooks/              # Jupyter notebooks for experimentation and training
├── src/                    # Source code for data processing and model training
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies


**Pipeline**
Data Preprocessing:
Text cleaning, tokenization, lemmatization, and stopword removal.
Feature Extraction:
TF-IDF Vectorization.
Model Training:
Various algorithms are trained and evaluated.
Evaluation:
Confusion matrix, accuracy score, and F1 score are calculated.
Visualization:
Comparative analysis of algorithm performance using visual plots.


**Results**
The model achieved an accuracy of over 85% using a multi-algorithm approach, with Random Forest and SVM performing best. Future work includes refining the feature extraction process and experimenting with deep learning models.
