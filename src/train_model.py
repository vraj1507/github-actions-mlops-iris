from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from datetime import datetime

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/iris_model_{timestamp}.pkl"
joblib.dump(model, model_path)

with open("models/latest_model_path.txt", "w") as f:
    f.write(model_path)