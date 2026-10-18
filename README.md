# Docker Lab 1: Containerized Model Training

This lab trains a Random Forest classifier on the Iris dataset inside a Docker container. The container evaluates the model on a held-out test set and writes the trained model and its metrics to a folder that is mounted from the host, so the files remain after the container exits.

## Project structure

```
.
├── dockerfile
├── .dockerignore
└── src/
    ├── main.py            # trains, evaluates and saves the model
    └── requirements.txt   # pinned dependencies
```

## Changes from the base lab

| Area | Base lab | This version |
|---|---|---|
| Base image | `python:3.10` | `python:3.10-slim` |
| Image size | 2.06 GB | 543 MB |
| Dependencies | unpinned | pinned (`scikit-learn==1.7.2`, `joblib==1.6.0`, `numpy==2.2.6`) |
| Layer caching | code and dependencies copied together | `requirements.txt` copied and installed first, so code changes don't trigger a reinstall |
| Evaluation | none | test accuracy plus a per-class classification report |
| Output | model saved inside the container and lost when it exits | `iris_model.pkl` and `metrics.json` written to a mounted volume |
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

Export the image as a tarball (git ignores the `.tar` file):

```sh
docker save lab1:v2 > my_image.tar
```

## Sample output

```
Train samples: 120, test samples: 30
Test accuracy: 0.9000

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.82      0.90      0.86        10
   virginica       0.89      0.80      0.84        10

    accuracy                           0.90        30
   macro avg       0.90      0.90      0.90        30
weighted avg       0.90      0.90      0.90        30

Model saved to /app/output/iris_model.pkl
The model training was successful
```

`output/metrics.json`:

```json
{
  "accuracy": 0.9,
  "n_estimators": 100,
  "test_size": 0.2,
  "random_state": 42,
  "sklearn_version": "1.7.2",
  "trained_at": "2026-10-18T03:23:36+00:00"
}
```

## Useful commands

```sh
docker images                 # list images and their sizes
docker ps -a                  # list containers, including stopped ones
docker rmi lab1:v2            # remove the image
```
