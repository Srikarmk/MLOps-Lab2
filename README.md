# Docker Lab 1: Containerized Model Training

This lab trains a Random Forest classifier on the Breast Cancer Wisconsin dataset (569 samples, 30 features, malignant vs. benign) inside a Docker container. The container evaluates the model on a held-out test set and writes the trained model and its metrics to a folder that is mounted from the host, so the files remain after the container exits.

## Project structure

```
.
├── dockerfile
├── .dockerignore
├── samples/
│   └── new_samples.csv    # 4 example rows for `predict`
└── src/
    ├── entrypoint.sh      # picks the mode: train / predict / any command
    ├── main.py            # trains, evaluates and saves the model
    ├── predict.py         # loads the saved model and scores new samples
    └── requirements.txt   # pinned dependencies
```

## Changes from the base lab

| Area | Base lab | This version |
|---|---|---|
| Base image | `python:3.10` | `python:3.10-slim` |
| Image size | 2.06 GB | 543 MB |
| Dependencies | unpinned | pinned (`scikit-learn==1.7.2`, `joblib==1.6.0`, `numpy==2.2.6`) |
| Layer caching | code and dependencies copied together | `requirements.txt` copied and installed first, so code changes don't trigger a reinstall |
| Dataset | Iris (150 samples, 3 classes) | Breast Cancer Wisconsin (569 samples, 30 features, binary) |
| Evaluation | none | test accuracy, ROC-AUC and a per-class classification report |
| Output | model saved inside the container and lost when it exits | `breast_cancer_model.pkl` and `metrics.json` written to a mounted volume |
| Configuration | hard-coded | `N_ESTIMATORS`, `TEST_SIZE`, `RANDOM_STATE`, `OUTPUT_DIR` environment variables |
| Train/test split | random | stratified, so every class appears in the test set |

## Running it

Build the image:

```sh
docker build -t lab1:v2 .
```

Run training and save the outputs to `./output` on the host:

```sh
docker run --rm -v "$(pwd)/output:/app/output" lab1:v2
```

Change hyperparameters without rebuilding:

```sh
docker run --rm -v "$(pwd)/output:/app/output" -e N_ESTIMATORS=200 -e TEST_SIZE=0.3 lab1:v2
```

Score new samples with the saved model (mount the same `output` folder so `predict` can find it):

```sh
# 5 random rows from the dataset, shown next to their true labels
docker run --rm -v "$(pwd)/output:/app/output" lab1:v2 predict

# your own CSV, mounted read-only
docker run --rm -v "$(pwd)/output:/app/output" -v "$(pwd)/samples:/app/input:ro" lab1:v2 predict /app/input/new_samples.csv
```

Predictions are printed and written to `output/predictions.csv`. Run `docker run --rm lab1:v2 help` to see every mode.

Export the image as a tarball (git ignores the `.tar` file):

```sh
docker save lab1:v2 > my_image.tar
```

## Sample output

```
Train samples: 455, test samples: 114, features: 30
Test accuracy: 0.9561
Test ROC-AUC:  0.9937

              precision    recall  f1-score   support

   malignant       0.95      0.93      0.94        42
      benign       0.96      0.97      0.97        72

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

Model saved to /app/output/breast_cancer_model.pkl
The model training was successful
```

`output/metrics.json`:

```json
{
  "accuracy": 0.9561,
  "roc_auc": 0.9937,
  "n_estimators": 100,
  "test_size": 0.2,
  "random_state": 42,
  "sklearn_version": "1.7.2",
  "trained_at": "2026-10-18T03:28:29+00:00"
}
```

## Useful commands

```sh
docker images                 # list images and their sizes
docker ps -a                  # list containers, including stopped ones
docker rmi lab1:v2            # remove the image
```
