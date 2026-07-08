import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from pathlib import Path
from src.preprocess import load_data, basic_preprocess

def train_and_save(data_path, model_out_path):
    df = load_data(data_path)
    X, y = basic_preprocess(df)

    # Simple fillna for numeric columns with median
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Save model and feature columns
    model_out_path = Path(model_out_path)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'columns': list(X.columns)}, model_out_path)
    print(f"Saved model to {model_out_path}")

if __name__ == '__main__':
    train_and_save('../data/heart_disease.csv', '../models/heart_model.pkl')
