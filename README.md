# Sentiment Analysis with Logistic Regression

This repository contains code for sentiment analysis using Logistic Regression on a movie review dataset. The project includes data loading, preprocessing, feature engineering using TF-IDF vectorization, model training, and performance evaluation.

## Overview

Sentiment analysis is a natural language processing task that involves classifying text into predefined categories based on the expressed sentiment. In this project, we use a Logistic Regression model to predict the sentiment of movie review phrases.

## Dataset

The dataset used for this project consists of two main files:
- `train.tsv`: Training data containing movie review phrases with corresponding sentiment labels.
- `test.tsv`: Test data for evaluating the trained model's performance.

## Technologies Used

- Python
- Pandas: Data manipulation and analysis.
- NLTK (Natural Language Toolkit): Text preprocessing, including tokenization, stemming, and stop word removal.
- Scikit-Learn: Machine learning library for TF-IDF vectorization and Logistic Regression model.

## Workflow

1. **Data Loading and Exploration**:
   - Read and explore the dataset using Pandas.
   - Check for missing values and understand the data structure.

2. **Text Preprocessing**:
   - Tokenize phrases, perform stemming, and remove stop words using NLTK.

3. **Feature Engineering**:
   - Use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical features.

4. **Model Training**:
   - Split the dataset into training and validation sets.
   - Train a Logistic Regression model on the TF-IDF transformed features.

5. **Model Evaluation**:
   - Evaluate the model's performance on both training and validation datasets using accuracy scores.
   - Optionally, visualize a confusion matrix to analyze prediction errors.

## Usage

Clone the repository and run the provided Jupyter notebook (`sentiment_analysis_logistic_regression.ipynb`) to reproduce the sentiment analysis task. Make sure to have Python and the required libraries installed.

```bash
git clone https://github.com/pragati9998/Sentimental-Analysis.git
cd Sentimental-Analysis
jupyter notebook sentiment_analysis_logistic_regression.ipynb
