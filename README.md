# Naive Bayes Text Classifier (From Scratch)

This project implements Naive Bayes classifiers **from scratch in Python** for text classification tasks, without using any machine learning libraries.  
The goal is to understand probabilistic modeling, likelihood estimation, and evaluation in supervised learning.

Two datasets are used:
- SMS Spam Collection (binary classification)
- BBC News Articles (multi-class classification)

---

## Datasets

### 1. SMS Spam Collection (UCI Repository)
- Task: Binary classification (spam vs ham)
- Total messages: 5,572
- Train–test split: 80% / 20%
- Short text messages, suitable for **Binary Naive Bayes**

### 2. BBC News Articles (Kaggle)
- Task: Multi-class classification  
  *(business, entertainment, politics, sport, tech)*
- Total articles: 2,225
- Stratified 80% / 20% train–test split
- Longer documents, suitable for **Multinomial Naive Bayes**

---

## Model Overview

The classifier is based on **Bayes’ Theorem**:

\[
P(C \mid d) \propto P(d \mid C) \cdot P(C)
\]

Where:
- \( P(C) \) is the prior probability of class \( C \)
- \( P(d \mid C) \) is the likelihood of document \( d \) given class \( C \)

---

## Implemented Variants

### Binary Naive Bayes (SMS Spam)
- Uses **presence/absence** of words
- Computes likelihood based on how many documents of a class contain a word
- Suitable for short, sparse text

### Multinomial Naive Bayes (BBC News)
- Uses **word frequency counts**
- Computes likelihood using token counts per class
- Suitable for longer documents

Both variants apply **Laplace smoothing (α = 1)** and operate in **log space** to avoid numerical underflow.

---

## Preprocessing

- Lowercasing
- Removal of punctuation
- Tokenization
- Digits retained for SMS spam detection
- Vocabulary built **only from training data**

Vocabulary sizes:
- SMS: ~8,463 words
- BBC News: ~28,463 words

---

## Evaluation

### SMS Spam (Binary Model)
- Test Accuracy: **97.8%**
- Confusion Matrix shows strong separation between spam and ham

### BBC News (Multinomial Model)
- Test Accuracy: **97.8%**
- Stratified evaluation across all five classes
- High diagonal dominance in confusion matrix

Confusion matrices are visualized using Matplotlib for interpretability.

---

## Analysis: Indicative Words

Top words per class were extracted from learned likelihoods to inspect model behavior.

- SMS spam messages show strong indicators such as *“free”*, *“call”*, and *“now”*
- BBC categories exhibit expected frequency-based patterns typical of Multinomial Naive Bayes

This highlights known characteristics and limitations of Naive Bayes on natural language data.

---

## How to Run

```bash
pip install pandas numpy matplotlib
python naive_bayes.py
