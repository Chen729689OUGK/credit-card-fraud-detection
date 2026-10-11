# Credit Card Fraud Detection

## Overview

This project explores the detection of fraudulent credit card transactions using machine learning techniques.

The objective is to analyze transaction data, address class imbalance, train classification models, and evaluate their ability to identify fraudulent transactions.

## Dataset

The project is designed for the Credit Card Fraud Detection dataset available on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymized transaction features, transaction amounts, timestamps, and a binary target variable named `Class`.

- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction

The dataset is not included in this repository.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

## Project Workflow

1. Load and inspect the transaction dataset.
2. Perform exploratory data analysis.
3. Examine missing values and class distribution.
4. Scale the transaction amount.
5. Split the data into training and test sets.
6. Train Decision Tree and Random Forest classifiers.
7. Evaluate the models using classification metrics and confusion matrices.
8. Experiment with SMOTE to address class imbalance.

## Evaluation Metrics

The project uses the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Precision, recall, and F1-score are particularly important because fraudulent transactions represent a small minority of the dataset.

## Repository Structure

```text
credit-card-fraud-detection/
├── credit_card_fraud.ipynb
├── credit_card_fraud.py
├── requirements.txt
├── .gitignore
└── README.md






