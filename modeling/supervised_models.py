"""
Supervised Models Module

This module contains supervised learning algorithms including Decision Tree
and Random Forest classifiers for customer segmentation prediction.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, 'modeling', 'pipeline_data')
LABELED_CSV = os.path.join(PIPELINE_DIR, 'data_labeled.csv')


def load_data(path=LABELED_CSV):
    """Load labeled customer data"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labeled data not found: {path}")
    return pd.read_csv(path)


def prepare_features(df, target='cluster'):
    """Prepare features and target for training"""
    # select numeric columns (exclude id and label cols)
    drop = {'customer_id', 'Cluster', 'Segment'}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c not in drop]
    X = df[features].copy()

    if target == 'cluster':
        y = df['Cluster'].values
    else:
        le = LabelEncoder()
        y = le.fit_transform(df['Segment'].astype(str).values)

    return X, y


def evaluate(model, X_train, X_test, y_train, y_test, cv=5):
    """Evaluate model performance"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    acc = accuracy_score(y_test, y_pred)
    scores = cross_val_score(model, pd.concat([X_train, X_test]), np.concatenate([y_train, y_test]), cv=cv)
    print(f"Accuracy (holdout): {acc:.4f}")
    print(f"CV ({cv}-fold) accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return acc, scores


def train_decision_tree(X, y, max_depth=None, cv=5):
    """Train and evaluate Decision Tree classifier"""
    print('\n--- Decision Tree ---')
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    dt_acc, dt_scores = evaluate(dt, X_train, X_test, y_train, y_test, cv=cv)
    return dt, dt_acc, dt_scores


def train_random_forest(X, y, n_estimators=100, cv=5):
    """Train and evaluate Random Forest classifier"""
    print('\n--- Random Forest ---')
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf_acc, rf_scores = evaluate(rf, X_train, X_test, y_train, y_test, cv=cv)
    return rf, rf_acc, rf_scores


def save_models(dt_model, rf_model, target='cluster', save_dir=None):
    """Save trained models to disk"""
    save_dir = save_dir or PIPELINE_DIR
    os.makedirs(save_dir, exist_ok=True)
    dt_path = os.path.join(save_dir, f"decision_tree_{target}.joblib")
    rf_path = os.path.join(save_dir, f"random_forest_{target}.joblib")
    joblib.dump(dt_model, dt_path)
    joblib.dump(rf_model, rf_path)
    print(f"\nSaved Decision Tree -> {dt_path}")
    print(f"Saved Random Forest -> {rf_path}")


def main(args):
    """Main training function"""
    df = load_data(args.labeled)
    X, y = prepare_features(df, target=args.target)
    print(f"Loaded {len(df)} rows, using {X.shape[1]} numeric features")

    # Decision Tree
    dt, dt_acc, dt_scores = train_decision_tree(X, y, max_depth=args.dt_max_depth, cv=args.cv)

    # Random Forest
    rf, rf_acc, rf_scores = train_random_forest(X, y, n_estimators=args.rf_estimators, cv=args.cv)

    # Save models
    save_models(dt, rf, target=args.target, save_dir=args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Decision Tree and Random Forest models')
    parser.add_argument('--labeled', type=str, default=LABELED_CSV, help='Path to data_labeled.csv')
    parser.add_argument('--target', choices=['cluster', 'segment'], default='cluster', help='Predict cluster or segment')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save trained models')
    parser.add_argument('--dt-max-depth', type=int, default=None, help='Max depth for decision tree')
    parser.add_argument('--rf-estimators', type=int, default=100, help='Number of trees for random forest')
    parser.add_argument('--cv', type=int, default=5, help='Cross-validation folds')
    args = parser.parse_args()
    main(args)