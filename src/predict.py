import csv
import os
import sys

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "breast_cancer_model.pkl")
N_DEMO_SAMPLES = int(os.getenv("N_DEMO_SAMPLES", "5"))


def load_samples(csv_path, feature_names):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [name for name in feature_names if name not in reader.fieldnames]
        if missing:
            sys.exit(f"Input CSV is missing {len(missing)} feature column(s), e.g. '{missing[0]}'")
        rows = [[float(row[name]) for name in feature_names] for row in reader]
    return np.array(rows), None


def demo_samples(data):
    rng = np.random.default_rng()
    idx = rng.choice(len(data.data), size=N_DEMO_SAMPLES, replace=False)
    return data.data[idx], data.target[idx]


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"No model found at {MODEL_PATH}. Run 'train' first with the same volume mounted.")

    model = joblib.load(MODEL_PATH)
    data = load_breast_cancer()

    if len(sys.argv) > 1:
        X, y_true = load_samples(sys.argv[1], data.feature_names)
        print(f"Loaded {len(X)} sample(s) from {sys.argv[1]}")
    else:
        X, y_true = demo_samples(data)
        print(f"No input file given, scoring {len(X)} random samples from the dataset")

    preds = model.predict(X)
    probas = model.predict_proba(X)

    out_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "prediction", "p_malignant", "p_benign"])
        for i, (pred, proba) in enumerate(zip(preds, probas)):
            label = data.target_names[pred]
            writer.writerow([i, label, f"{proba[0]:.3f}", f"{proba[1]:.3f}"])
            line = f"  sample {i}: {label:<9}  (p_malignant={proba[0]:.3f})"
            if y_true is not None:
                line += f"  actual: {data.target_names[y_true[i]]}"
            print(line)

    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
