# Credit Card Fraud Detection – IPD Prototype

This project is an Interim Progress Demo (IPD) prototype for a credit card fraud detection system using machine learning.  
It demonstrates a complete pipeline from loading a trained model to generating fraud predictions through a simple web interface.

## What the program does
- Loads a pre-trained **Random Forest** model and a fitted **scaler**
- Accepts transaction data as a CSV file
- Validates and scales input features
- Predicts **fraud probability** for each transaction
- Applies a user-controlled threshold to classify transactions
- Displays results and allows downloading predictions as a CSV file

## Files in this repository
- `app.py` – Streamlit application (user interface + prediction logic)
- `fraud_model.pkl` – trained machine learning model
- `scaler.pkl` – fitted feature scaler

## Dataset (not included)
The full dataset is not included because it exceeds GitHub’s file size limit.

**Dataset name:** Credit Card Fraud Detection (European cardholders, 2013)  
**Download link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### How to get the dataset
1. Open the link above and download the dataset from Kaggle.
2. Unzip the file.
3. Upload on the prototype
