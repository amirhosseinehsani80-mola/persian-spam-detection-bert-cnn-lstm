# Persian Spam Email Detection Using ParsBERT and Deep Learning Models

This repository contains a complete implementation of Persian spam email classification using modern natural language processing and deep learning techniques. The project includes a full preprocessing pipeline, feature extraction using ParsBERT, dimensionality reduction with PCA, multiple neural network architectures, and classical machine learning baselines. The goal is to compare different modeling approaches and evaluate their effectiveness on the Persian Spam Email dataset.

## Overview

The project focuses on building a binary classifier capable of distinguishing between spam and non-spam Persian emails. The workflow covers dataset acquisition, cleaning, embedding generation, model development, training, and performance evaluation. The final results compare deep learning models with traditional machine learning techniques.
<img width="700" height="477" alt="image" src="https://github.com/user-attachments/assets/85bd4a14-5c25-4299-9df4-435b2f618361" />

## Data Preprocessing

The text preprocessing pipeline includes:
- Removal of URLs, email addresses, and phone numbers
- Reduction of repeated characters
- Removal of Persian stop words
- Tokenization using the ParsBERT tokenizer
- Padding of token sequences for uniform length

This ensures clean and consistent input for the embedding model and downstream classifiers.

## ParsBERT Embeddings and PCA

ParsBERT is used to generate contextual embeddings for each email. These high-dimensional embeddings are then flattened and reduced using PCA to 120 components. This significantly reduces computational requirements while preserving meaningful semantic information.

## Deep Learning Models

Three deep learning models are trained and evaluated:

### CNN-LSTM Model
A hybrid network combining:
- Convolutional layers for pattern extraction  
- LSTM layers for sequential representation  
- Dense layers with dropout and batch normalization  

A hyperparameter search across batch size, learning rate, and optimizer type identifies the best configuration.

### CNN Model
A convolutional neural network trained on PCA-reduced embeddings. This model focuses on local feature extraction and performs well on structured embeddings.

### LSTM Model
A pure LSTM architecture trained to capture sequential dependencies in the reduced embedding sequences.

Each model is trained with early stopping and evaluated on a held-out test set.

## Classical Machine Learning Baselines

To compare deep learning with traditional approaches, a Bag-of-Words representation is created using CountVectorizer. The following models are trained:

- Multinomial Naive Bayes  
- Logistic Regression  
- Support Vector Machine  
- Random Forest  

Each model is evaluated using accuracy, precision, recall, and F1-score.

## Model Comparison
### Model Performance Comparison

| Model               | Accuracy | Precision | Recall   | F1-Score |
|---------------------|----------|-----------|----------|----------|
| CNN-LSTM            | 0.953333 | 0.965753  | 0.940000 | 0.952703 |
| CNN                 | 0.963333 | 0.972789  | 0.953333 | 0.962963 |
| LSTM                | 0.500000 | 0.500000  | 0.053333 | 0.096386 |
| Naive Bayes         | 0.976667 | 0.993103  | 0.960000 | 0.976271 |
| Logistic Regression | 0.963333 | 0.979310  | 0.946667 | 0.962712 |
| SVM                 | 0.883333 | 0.913669  | 0.846667 | 0.878893 |
| Random Forest       | 0.940000 | 0.940000  | 0.940000 | 0.940000 |

                
A combined performance table is constructed comparing:
- CNN-LSTM  
- CNN  
- LSTM  
- Naive Bayes  
- Logistic Regression  
- SVM  
- Random Forest  

This comparison highlights the strengths and weaknesses of deep learning versus classical machine learning for Persian spam detection.

## Summary

This project demonstrates a complete NLP pipeline for Persian text classification. It covers preprocessing, embedding generation with ParsBERT, dimensionality reduction, deep learning architectures, classical ML methods, and thorough evaluation. The results show the impact of model architecture and representation choice on spam detection accuracy.
