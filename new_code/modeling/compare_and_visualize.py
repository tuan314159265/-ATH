"""
Compare Decision Tree and Random Forest to K-Means clusters and visualize results.

Produces and saves:
- confusion matrices (DT and RF)
- feature importances (top features)
- PCA 2D scatter: true cluster vs DT vs RF predictions
- silhouette score for clusters

Outputs are saved to `outputs/model_comparison/` as PNGs.

Run:
  python3 modeling/compare_and_visualize.py
  python3 modeling/compare_and_visualize.py --target cluster --save-dir outputs/model_comparison
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, 'modeling', 'pipeline_data')
LABELED_CSV = os.path.join(PIPELINE_DIR, 'data_labeled.csv')
SCALED_CSV = os.path.join(PIPELINE_DIR, 'rfm_scaled.csv')


def load_df(path=LABELED_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labeled CSV not found: {path}")
    return pd.read_csv(path)


def prepare_X_y(df, target='cluster'):
    drop = {'customer_id', 'Cluster', 'Segment'}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c not in drop]
    X = df[features].copy()
    if target == 'cluster':
        y = df['Cluster'].values
    else:
        le = LabelEncoder()
        y = le.fit_transform(df['Segment'].astype(str).values)
    return X, y, features


def plot_confusion(y_true, y_pred, classes, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importances(importances, features, title, out_path, top_n=10):
    idx = np.argsort(importances)[-top_n:][::-1]
    top_feats = [features[i] for i in idx]
    top_vals = importances[idx]
    plt.figure(figsize=(8,4))
    sns.barplot(x=top_vals, y=top_feats)
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pca_compare(X, y_true, y_dt, y_rf, classes, out_base):
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax, lab, col in zip(axes, [y_true, y_dt, y_rf], ['True', 'DT', 'RF']):
        sc = ax.scatter(Z[:,0], Z[:,1], c=lab, cmap='tab10', s=10)
        ax.set_title(col)
    plt.suptitle('PCA projection: True vs DT vs RF')
    plt.tight_layout(rect=[0,0,1,0.95])
    out_path = os.path.join(out_base, 'pca_compare.png')
    plt.savefig(out_path)
    plt.close()


def main(args):
    out_dir = args.save_dir or os.path.join(PROJECT_ROOT, 'outputs', 'model_comparison')
    os.makedirs(out_dir, exist_ok=True)

    df = load_df(args.labeled)
    X, y, features = prepare_X_y(df, target=args.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train DT and RF
    dt = DecisionTreeClassifier(max_depth=args.dt_max_depth, random_state=42)
    rf = RandomForestClassifier(n_estimators=args.rf_estimators, random_state=42, n_jobs=-1)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_dt = dt.predict(X_test)
    y_rf = rf.predict(X_test)

    print('Decision Tree report:')
    print(classification_report(y_test, y_dt, zero_division=0))
    print('Random Forest report:')
    print(classification_report(y_test, y_rf, zero_division=0))

    classes = np.unique(y)

    # confusion matrices
    plot_confusion(y_test, y_dt, classes, 'Decision Tree Confusion', os.path.join(out_dir, 'confusion_dt.png'))
    plot_confusion(y_test, y_rf, classes, 'Random Forest Confusion', os.path.join(out_dir, 'confusion_rf.png'))

    # feature importances (RF & DT)
    plot_feature_importances(rf.feature_importances_, features, 'Random Forest Importances', os.path.join(out_dir, 'feat_imp_rf.png'))
    try:
        plot_feature_importances(dt.feature_importances_, features, 'Decision Tree Importances', os.path.join(out_dir, 'feat_imp_dt.png'))
    except Exception:
        pass

    # PCA compare plot on test set
    X_test_full = pd.concat([X_train, X_test]).reset_index(drop=True)
    # create predictions aligned with X_test_full
    y_full_dt = np.concatenate([dt.predict(X_train), dt.predict(X_test)])
    y_full_rf = np.concatenate([rf.predict(X_train), rf.predict(X_test)])
    y_full_true = np.concatenate([y_train, y_test])
    plot_pca_compare(X_test_full.values, y_full_true, y_full_dt, y_full_rf, classes, out_dir)

    # silhouette (use scaled data if available)
    if os.path.exists(SCALED_CSV):
        X_scaled = pd.read_csv(SCALED_CSV)
        if 'customer_id' in X_scaled.columns and 'customer_id' in df.columns:
            merged = pd.merge(df[['customer_id', 'Cluster']], X_scaled, on='customer_id', how='left')
            labels = merged['Cluster'].values
            X_for_sil = merged.drop(columns=['customer_id', 'Cluster']).values
            try:
                s = silhouette_score(X_for_sil, labels)
                print(f'Silhouette score (clusters): {s:.4f}')
            except Exception as e:
                print('Silhouette score error:', e)
        else:
            try:
                s = silhouette_score(X.values, df['Cluster'].values)
                print(f'Silhouette score (clusters): {s:.4f}')
            except Exception as e:
                print('Silhouette score error:', e)
    else:
        try:
            s = silhouette_score(X.values, df['Cluster'].values)
            print(f'Silhouette score (clusters): {s:.4f}')
        except Exception as e:
            print('Silhouette score error:', e)

    # save models
    joblib.dump(dt, os.path.join(out_dir, 'decision_tree.joblib'))
    joblib.dump(rf, os.path.join(out_dir, 'random_forest.joblib'))
    print('Saved plots and models to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled', type=str, default=LABELED_CSV)
    parser.add_argument('--target', choices=['cluster', 'segment'], default='cluster')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--dt-max-depth', type=int, default=None)
    parser.add_argument('--rf-estimators', type=int, default=100)
    args = parser.parse_args()
    main(args)
