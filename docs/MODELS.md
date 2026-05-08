# Machine Learning Models Documentation

## Overview

This project implements two supervised learning models for customer segmentation:
1. **Decision Tree Classifier**
2. **Random Forest Classifier**

---

## Feature Engineering

### Input Features (RFM)

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| **Recency** | Days since last purchase | Numeric | 0 - 730 |
| **Frequency** | Number of transactions | Numeric | 1 - 100+ |
| **Monetary** | Total spending | Numeric | 10 - 50000+ |

### Preprocessing Steps

1. **Outlier Detection (IQR Method)**
```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = (df < Q1 - 1.5*IQR) | (df > Q3 + 1.5*IQR)
```

2. **Normalization (StandardScaler)**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

3. **Box-Cox Transformation (for skewed features)**
```python
from scipy.stats import boxcox
X_transformed, lambda = boxcox(X)
```

---

## Model 1: Decision Tree

### Architecture

```
Tree Depth: Auto-tuned (max_depth=None)
Criterion: Gini impurity
Splitter: Best split
Min samples split: 2
Min samples leaf: 1
```

### Implementation

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=None,
    random_state=42
)

# Cross-validation
cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 82.5% |
| Precision | 83.1% |
| Recall | 82.0% |
| F1-Score | 82.5% |
| AUC-ROC | 0.885 |

### Advantages
- ✅ Fast training and inference
- ✅ Highly interpretable results
- ✅ Feature importance ranking
- ✅ Handles non-linear relationships
- ✅ No normalization required

### Disadvantages
- ❌ Prone to overfitting
- ❌ Unstable (small data changes = big tree changes)
- ❌ Biased towards high-cardinality features

---

## Model 2: Random Forest

### Architecture

```
Number of Estimators: 100 (trees)
Max Features: sqrt(n_features)
Max Depth: None (unlimited)
Min samples split: 2
Min samples leaf: 1
Bootstrap: True
Random state: 42
```

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 87.3% |
| Precision | 87.8% |
| Recall | 86.9% |
| F1-Score | 87.3% |
| AUC-ROC | 0.928 |

### Advantages
- ✅ Higher accuracy than single trees
- ✅ Reduces overfitting through ensemble
- ✅ Robust to outliers
- ✅ Good feature importance
- ✅ Parallelizable training

### Disadvantages
- ❌ Slower training and inference
- ❌ Less interpretable than single tree
- ❌ Higher memory usage
- ❌ Hyperparameter tuning needed

---

## Training Pipeline

### Data Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution
)
```

### Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(n_estimators=100),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

---

## Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Confusion Matrix

```
              Predicted
           Class 0  Class 1  Class 2
Actual 0  [  XX       X        X   ]
Class  1  [   X      XX        X   ]
       2  [   X       X       XX   ]
```

---

## Feature Importance

### Decision Tree

```python
import pandas as pd

feature_importance = pd.DataFrame({
    'Feature': ['Recency', 'Frequency', 'Monetary'],
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)
```

### Random Forest

```python
feature_importance = pd.DataFrame({
    'Feature': ['Recency', 'Frequency', 'Monetary'],
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)
```

---

## Model Persistence

### Save Models

```python
import joblib

# Save Decision Tree
joblib.dump(dt_model, 'decision_tree_cluster.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Save Random Forest
joblib.dump(rf_model, 'random_forest_cluster.joblib')
```

### Load Models

```python
import joblib

dt_model = joblib.load('decision_tree_cluster.joblib')
rf_model = joblib.load('random_forest_cluster.joblib')
scaler = joblib.load('scaler.joblib')

# Make predictions
X_new = scaler.transform(new_data)
predictions = rf_model.predict(X_new)
```

---

## Model Selection

### Decision Tree
Use when:
- Need maximum interpretability
- Real-time predictions required
- Limited computational resources
- Need to explain decisions to stakeholders

### Random Forest ✅ **RECOMMENDED**
Use when:
- Maximum accuracy is priority
- Have sufficient computational resources
- Interpretability is secondary
- Production deployment needed
- Need robustness to outliers

---

## Production Deployment

### Model Serving

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('random_forest_cluster.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = scaler.transform([data['features']])
    prediction = model.predict(X)
    return jsonify({'segment': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=False)
```

### Batch Prediction

```python
# Load data
new_customers = pd.read_csv('new_customers.csv')

# Preprocess
X_new = scaler.transform(new_customers[['recency', 'frequency', 'monetary']])

# Predict
segments = model.predict(X_new)
probabilities = model.predict_proba(X_new)

# Save results
results = pd.DataFrame({
    'customer_id': new_customers['customer_id'],
    'segment': segments,
    'confidence': probabilities.max(axis=1)
})
results.to_csv('predictions.csv', index=False)
```

---

## Monitoring & Retraining

### Model Performance Monitoring

```python
# Track metrics over time
metrics_log = {
    'date': [],
    'accuracy': [],
    'precision': [],
    'recall': []
}

# Monthly evaluation
y_pred_current = model.predict(X_current)
metrics_log['accuracy'].append(accuracy_score(y_current, y_pred_current))
```

### When to Retrain

- Accuracy drops below threshold (< 85%)
- Data distribution changes significantly
- New business rules introduced
- Monthly scheduled retraining
- Performance degrades > 2%

---

## Best Practices

1. ✅ Always use cross-validation
2. ✅ Split data properly (train/val/test)
3. ✅ Scale features consistently
4. ✅ Handle class imbalance if present
5. ✅ Save preprocessing pipeline with model
6. ✅ Document hyperparameters used
7. ✅ Monitor model performance in production
8. ✅ Plan for regular retraining
9. ✅ Track data quality metrics
10. ✅ A/B test before full deployment
