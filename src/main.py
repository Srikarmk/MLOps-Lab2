import json
import os
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "100"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=iris.target,
    )
    print(f"Train samples: {len(X_train)}, test samples: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "iris_model.pkl")
    joblib.dump(model, model_path)

    metrics = {
        "accuracy": round(accuracy, 4),
        "n_estimators": N_ESTIMATORS,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {model_path}")
    print("The model training was successful")


if __name__ == "__main__":
    main()
