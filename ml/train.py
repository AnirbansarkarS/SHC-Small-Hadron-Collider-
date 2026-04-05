import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from pathlib import Path

# ── Load ──────────────────────────────────────────────
NROWS = 100_000   # start here, increase later if you want

col_names = ['label',
             'lepton_pt', 'lepton_eta', 'lepton_phi',
             'missing_energy_magnitude', 'missing_energy_phi',
             'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_btag',
             'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_btag',
             'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_btag',
             'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_btag',
             'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

def train():
    csv_path = Path('data/datasets/HIGGS.csv')
    if not csv_path.exists():
        print(f"Error: Dataset not found at {csv_path}. Please download it first.")
        return

    df = pd.read_csv(csv_path,
                     header=None,
                     names=col_names,
                     nrows=NROWS)

    print(f"Loaded {len(df):,} rows")
    print(f"Signal:     {(df.label==1).sum():,}")
    print(f"Background: {(df.label==0).sum():,}")

    # ── Features ───────────────────────────────────────────
    # Use only high-level physics features (cols 22-28)
    # These are closest to what your simulation computes
    FEATURES = ['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

    X = df[FEATURES]
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # ── Model 1: Logistic Regression (baseline) ────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_sc, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:,1])
    print(f"\nLogistic Regression AUC: {lr_auc:.4f}")

    # ── Model 2: Random Forest ─────────────────────────────
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
    print(f"Random Forest AUC:       {rf_auc:.4f}")
    print(classification_report(y_test, rf.predict(X_test),
                                 target_names=['background','signal']))

    # ── Feature importance ─────────────────────────────────
    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    print("\nTop features:")
    print(importances.sort_values(ascending=False))

    os.makedirs('ml', exist_ok=True)
    # ── Save model ─────────────────────────────────────────
    joblib.dump(rf, 'ml/model.pkl')
    joblib.dump(scaler, 'ml/scaler.pkl')
    print("\nModel saved to ml/model.pkl")

if __name__ == "__main__":
    train()
