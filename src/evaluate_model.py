from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json
import os
from datetime import datetime

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with open("models/latest_model_path.txt", "r") as f:
    model_path = f.read().strip()

model = joblib.load(model_path)
preds = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, preds)),
    "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
    "model_path": model_path
}

os.makedirs("metrics", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"metrics/metrics_{timestamp}.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("metrics/latest_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)