# Classification, Decision Trees, and Clustering

## Objective
A set of supervised and unsupervised Machine Learning projects covering credit card fraud detection, medical classification, and customer segmentation.

## Content

### 1. Credit Card Fraud Detection — Random Forest
- Dataset: [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- `RandomForestClassifier` model trained on highly imbalanced data
- Evaluation via `classification_report` and confusion matrix — 76% recall for the minority class (fraud) with 94% precision

### 2. Drug Classification — Decision Tree
- Dataset: [Drugs A/B/C/X/Y](https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees)
- Categorical variable encoding (One-Hot Encoding) + `DecisionTreeClassifier`
- Model serialized using `pickle` for reuse (see prediction project below)

### 3. Prediction with Saved Model
- Loading the trained `.pkl` model and applying it to new data — a practical example of simple model deployment

### 4. Customer Segmentation — K-Means
- Dataset: [Credit Card Customer Attrition](https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m)
- Data normalization and application of the elbow method (inertia) to select the optimal number of clusters

## Technologies
`Python` `scikit-learn` `pandas` `pickle` `matplotlib`

## How to Run
```bash
pip install scikit-learn pandas numpy matplotlib kaggle
jupyter notebook "analiza fraud karty kredytowe - random forest.ipynb"
jupyter notebook "leki kargiologiczne i drzewo decyzyjne.ipynb"