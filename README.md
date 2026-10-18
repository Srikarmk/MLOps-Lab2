# Docker Lab 1: Containerized Model Training

This lab packages a machine learning training job into a Docker image. One image trains a Random Forest classifier on the Breast Cancer Wisconsin dataset, evaluates it, saves the model to a Docker volume, and runs predictions with the saved model in a separate container. A script runs the whole flow and also checks that the image can be exported to a file and loaded back with the same results.

You don't need Python or any ML libraries on your machine. Everything runs inside the container.

- [What you need installed](#what-you-need-installed)
- [Quick start](#quick-start)
- [What a successful run looks like](#what-a-successful-run-looks-like)
- [Running each step by hand](#running-each-step-by-hand)
- [Changes made to the base lab](#changes-made-to-the-base-lab)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Cleaning up](#cleaning-up)

## What you need installed

| Tool | Version | Why |
|---|---|---|
| Docker Desktop (macOS / Windows) or Docker Engine (Linux) | Tested with Docker Desktop 4.94.0, Engine 29.8.2. Any Engine 24+ should work. | Builds and runs the image |
| Git | Any recent version | Clones this repository |
| Disk space | About 1.5 GB free | The base image, the built image (543 MB) and the exported tarball (111 MB) |

No Python, pip or scikit-learn is needed on the host. The image comes with Python 3.10 and pinned versions of every library.

### 1. Install Docker

- **macOS**: download Docker Desktop from https://docs.docker.com/desktop/setup/install/mac-install/ (pick Apple Silicon or Intel), open the `.dmg`, drag Docker to Applications, then launch it once and accept the prompts.
- **Windows**: install WSL 2 first by running `wsl --install` in an admin PowerShell and restarting. Then install Docker Desktop from https://docs.docker.com/desktop/setup/install/windows-install/ with the "Use WSL 2" option checked. Run every command in this README from a **WSL (Ubuntu) terminal**, not PowerShell.
- **Linux (Ubuntu)**: follow https://docs.docker.com/engine/install/ubuntu/, then let your user run Docker without `sudo`:
  ```sh
  sudo usermod -aG docker $USER   # log out and back in afterwards
  ```

### 2. Check that Docker is running

Docker Desktop has to be open (the whale icon is in the menu bar or system tray). Then run:

```sh
docker --version
docker info --format '{{.ServerVersion}}'
```

The first command prints something like `Docker version 29.8.2`. The second prints the Engine version. If it prints an error about the "docker daemon" or `docker.sock`, Docker Desktop isn't running yet. Start it and wait about 10 seconds.

### 3. Get the code

```sh
git clone https://github.com/Srikarmk/MLOps-Lab2.git
cd MLOps-Lab2
```

## Quick start

One command builds the image and runs every step:

```sh
./run_lab.sh
```

The first run downloads the `python:3.10-slim` base image and installs the libraries, which takes 1–3 minutes depending on your connection. Later runs take about 10 seconds because Docker caches the layers.

To remove everything the script created when it finishes (the containers, image, volume and tarball), run it with `--clean`. The `./output` folder is kept.

```sh
./run_lab.sh --clean
```

## What a successful run looks like

The script prints a `==>` header for each step. Docker's build log is left out below, and numbers like image IDs will differ on your machine.

```
==> Checking Docker
Client 29.8.2, Engine 29.8.2
==> Removing containers left from a previous run
==> Building lab1:latest
Image size: 543MB
==> Creating named volume lab1-output
Mountpoint inside the Docker VM: /var/lib/docker/volumes/lab1-output/_data
==> Baseline training run (temporary container, removed on exit with --rm)
Test accuracy: 0.9561
Test ROC-AUC:  0.9937
==> Saving the image to lab1-image.tar.gz
Tarball size: 111M
==> Removing lab1:latest locally and loading it back from lab1-image.tar.gz
lab1:latest removed
Loaded image: lab1:latest
==> Training with the loaded image (container lab1-train, kept after it exits)
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
==> Predicting (container lab1-predict reads the model from the same volume)
Loaded 4 sample(s) from /app/input/new_samples.csv
  sample 0: malignant  (p_malignant=0.970)
  sample 1: benign     (p_malignant=0.030)
  sample 2: malignant  (p_malignant=1.000)
  sample 3: malignant  (p_malignant=1.000)
Predictions saved to /app/output/predictions.csv
==> Checking the save/load round trip
Image ID before save: sha256:c4da244c6fc5...
Image ID after load:  sha256:c4da244c6fc5...
Accuracy before: 0.9561, after: 0.9561
Round trip OK: the loaded image is identical and reproduces the same result.
==> Copying the volume contents to ./output
-rw-r--r--  1 user  staff  324345  breast_cancer_model.pkl
-rw-r--r--  1 user  staff     185  metrics.json
-rw-r--r--  1 user  staff     137  predictions.csv
==> Containers kept for inspection
NAMES          IMAGE         STATUS
lab1-predict   lab1:latest   Exited (0) Less than a second ago
lab1-train     lab1:latest   Exited (0) Less than a second ago
View their output with: docker logs lab1-train   (or open them in Docker Desktop)
==> Done
```

A run succeeded if:

- it ends with `Round trip OK` and `==> Done`, and the exit code is 0 (`echo $?` prints `0`)
- test accuracy is **0.9561** and ROC-AUC is **0.9937**. The split and the model use a fixed random seed, so these match on every machine.
- the 4 rows in `samples/new_samples.csv` are predicted as malignant, benign, malignant, malignant, which are their true labels
- `./output` holds `breast_cancer_model.pkl`, `metrics.json` and `predictions.csv`
- Docker Desktop shows the containers `lab1-train` and `lab1-predict` as **Exited (0)**, the image `lab1:latest` and the volume `lab1-output`

"Exited" is the expected state. These containers run a single job and stop, unlike a web server that keeps running.

## Running each step by hand

These are the same steps the script runs, one command at a time.

**Build the image**

```sh
docker build -t lab1:latest .
```

**Create a named volume** to hold the model between containers:

```sh
docker volume create lab1-output
```

**Train.** With no arguments the container uses its default command, `train`:

```sh
docker run --name lab1-train -v lab1-output:/app/output lab1:latest
```

**Predict** with the saved model. Without a file it scores 5 random rows from the dataset and shows each one's true label:

```sh
docker run --rm -v lab1-output:/app/output lab1:latest predict
```

To score your own CSV (one column for each of the 30 features, with the same column names as `samples/new_samples.csv`), mount its folder read-only:

```sh
docker run --name lab1-predict -v lab1-output:/app/output -v "$(pwd)/samples:/app/input:ro" \
    lab1:latest predict /app/input/new_samples.csv
```

**Copy the results out** of the volume to your machine:

```sh
docker cp lab1-predict:/app/output/. ./output/
cat output/metrics.json
```

**Change hyperparameters** without rebuilding the image:

```sh
docker run --rm -v lab1-output:/app/output -e N_ESTIMATORS=300 -e TEST_SIZE=0.3 lab1:latest train
```

**Export the image to a file and load it back**, which is how you'd move it to a machine with no internet access:

```sh
docker save lab1:latest | gzip > lab1-image.tar.gz
docker rm lab1-train lab1-predict
docker rmi lab1:latest
gunzip -c lab1-image.tar.gz | docker load
```

**Other commands**

```sh
docker run --rm lab1:latest help                  # list the container's modes
docker run --rm lab1:latest python --version      # any other command passes straight through
docker run --rm -it lab1:latest sh                # open a shell inside the container
docker logs lab1-train                            # output of a finished container
docker inspect lab1:latest --format '{{json .Config.Labels}}'
```

## Changes made to the base lab

The base lab (`Labs/Docker_Labs/Lab1` in the course repo) is a one-stage `python:3.10` image that runs `main.py` to train on Iris and saves the model inside the container, where it's lost when the container exits. Its README had three commands. The changes below are mostly about how the image is built and how containers are run.

| # | Area | Base lab | This version | What it does |
|---|---|---|---|---|
| 1 | Base image | `python:3.10` | `python:3.10-slim` | Removes compilers and system packages the job never uses. The image drops from **2.06 GB to 543 MB**. |
| 2 | Build stages | one stage | `builder` stage + `runtime` stage | The builder installs the libraries into a virtual environment at `/opt/venv`, then removes `pip` and `setuptools` from it. The runtime stage copies only that finished environment and the app code. With these libraries the size stays at 543 MB because they all come as prebuilt wheels. The split starts saving space as soon as a dependency needs compiling, because the compilers stay in the builder stage. |
| 3 | Layer caching | code and requirements copied together | `requirements.txt` copied and installed before the code | Editing `main.py` only rebuilds the last small layer instead of reinstalling all the libraries. |
| 4 | Dependency versions | unpinned | `scikit-learn==1.7.2`, `joblib==1.6.0`, `numpy==2.2.6` | Every build installs the same versions, so results don't change when a new release comes out. |
| 5 | User | root | `appuser` (uid 1000) | If the code is compromised, it doesn't have root inside the container. `docker run --rm lab1:latest whoami` prints `appuser`. |
| 6 | Build argument | none | `ARG PYTHON_VERSION=3.10` | `docker build --build-arg PYTHON_VERSION=3.11 .` builds on another Python version without editing the file. |
| 7 | Labels | none | OCI labels (title, description, author, source) | Metadata about the image, shown by `docker inspect`. |
| 8 | Start command | `CMD ["python", "main.py"]` | `ENTRYPOINT` script plus `CMD ["train"]` | One image does several jobs: `train`, `predict`, `help`, or any other command. See [How it works](#how-it-works). |
| 9 | Prediction | none | `predict` mode (`src/predict.py`) | Loads the saved model and scores random dataset rows or a CSV that you mount in. Writes `predictions.csv`. |
| 10 | Where output goes | inside the container, lost when it exits | `/app/output`, declared as a `VOLUME` and backed by the named volume `lab1-output` | The model outlives the container that trained it, and a second container can read it. |
| 11 | Configuration | hard-coded | `N_ESTIMATORS`, `TEST_SIZE`, `RANDOM_STATE`, `OUTPUT_DIR`, `N_DEMO_SAMPLES` environment variables | Change settings with `-e` on `docker run` instead of rebuilding. |
| 12 | Build context | everything | `.dockerignore` | Keeps `.git`, docs, samples and tarballs out of what gets sent to `docker build`. |
| 13 | Export and import | `docker save` command only | `run_lab.sh` saves, deletes, reloads and checks the image | Confirms that the exported file is a complete copy: same image ID, same accuracy. |
| 14 | Automation | 3 commands in the README | `run_lab.sh` | Runs the whole lab with one command and stops with a clear message on any failure (Docker missing, daemon not running, round-trip mismatch). |
| 15 | Dataset and metrics | Iris (150 samples, 3 classes), no evaluation | Breast Cancer Wisconsin (569 samples, 30 features, binary). Reports accuracy, ROC-AUC and a classification report, saved to `metrics.json`. | A larger, more realistic dataset, with a stratified split so both classes appear in the test set. |

## How it works

### Project structure

```
.
├── dockerfile              # two-stage build: builder -> runtime
├── .dockerignore           # files kept out of the build context
├── .gitattributes          # keeps shell scripts with Unix line endings on Windows
├── run_lab.sh              # runs the whole lab end to end
├── samples/
│   └── new_samples.csv     # 4 example rows for `predict`
└── src/
    ├── entrypoint.sh       # picks the mode: train / predict / help / any command
    ├── main.py             # trains, evaluates and saves the model
    ├── predict.py          # loads the saved model and scores new samples
    └── requirements.txt    # pinned dependencies
```

### The build

```
builder stage (python:3.10-slim)          runtime stage (python:3.10-slim)
  python -m venv /opt/venv                  useradd appuser
  pip install -r requirements.txt   ---->   COPY --from=builder /opt/venv
  pip uninstall pip setuptools              COPY src/ /app
                                            USER appuser
                                            ENTRYPOINT ["sh", "/app/entrypoint.sh"]
                                            CMD ["train"]
```

Only the runtime stage becomes the final image. The builder stage exists only during the build.

### ENTRYPOINT and CMD

`ENTRYPOINT` always runs. `CMD` is only its default argument, and anything typed after the image name in `docker run` replaces it.

| You run | Container runs |
|---|---|
| `docker run lab1:latest` | `sh entrypoint.sh train` (the default CMD) |
| `docker run lab1:latest predict` | `sh entrypoint.sh predict` |
| `docker run lab1:latest predict /app/input/x.csv` | `sh entrypoint.sh predict /app/input/x.csv` |
| `docker run lab1:latest python --version` | `sh entrypoint.sh python --version`, which runs `python --version` |

### Volumes

```
             lab1-output (named volume, managed by Docker)
              ^                               |
  writes model |                               | reads model
              |                               v
   [ lab1-train container ]        [ lab1-predict container ] <-- samples/ (bind mount, read-only)
```

- A **named volume** (`-v lab1-output:/app/output`) is storage that Docker manages. It exists apart from any container and stays until `docker volume rm`. Docker fills a new volume with whatever the image has at that path, including the `appuser` ownership, so the non-root user can write to it on any OS.
- A **bind mount** (`-v "$(pwd)/samples:/app/input:ro"`) maps a folder on your machine into the container. `:ro` makes it read-only, so the container can read your CSV but can't change it.
- `docker cp` copies files out of the volume through a container. This works the same on macOS, Windows and Linux.

### Configuration

| Variable | Default | Used by |
|---|---|---|
| `N_ESTIMATORS` | `100` | `train`: number of trees |
| `TEST_SIZE` | `0.2` | `train`: share of the data held out for testing |
| `RANDOM_STATE` | `42` | `train`: seed for the split and the model |
| `OUTPUT_DIR` | `/app/output` | both: where the model, metrics and predictions go |
| `N_DEMO_SAMPLES` | `5` | `predict` with no file: number of random rows to score |

`run_lab.sh` also reads `IMAGE` (default `lab1:latest`) and `VOLUME` (default `lab1-output`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to the Docker daemon` / `failed to connect to the docker API` | Docker Desktop isn't running. Open it, wait for the whale icon to stop animating, and try again. |
| `docker: command not found` on macOS, even though Docker Desktop is installed | Docker Desktop puts the CLI in `~/.docker/bin`. Run `export PATH="$HOME/.docker/bin:$PATH"`, or add that line to `~/.zshrc`. |
| `permission denied while trying to connect to the Docker daemon socket` (Linux) | Run `sudo usermod -aG docker $USER`, then log out and back in. |
| `./run_lab.sh: Permission denied` | Run `chmod +x run_lab.sh`, or use `bash run_lab.sh`. |
| `No model found at /app/output/breast_cancer_model.pkl` | Run `train` first, and make sure `predict` mounts the same volume (`-v lab1-output:/app/output`). |
| `Conflict. The container name "/lab1-train" is already in use` | You're rerunning the manual steps. Run `docker rm lab1-train lab1-predict` first (`run_lab.sh` does this for you). |
| `Permission denied` writing to a bind-mounted `./output` folder on Linux | The container runs as uid 1000, and your user may have a different uid. Use the named volume and `docker cp` as above, or add `--user "$(id -u):$(id -g)"` to `docker run`. |
| `/app/entrypoint.sh: 2: set: Illegal option -` on Windows | The script was checked out with Windows line endings. Clone the repo from inside WSL, or run `git config --global core.autocrlf input` and clone again. |

## Cleaning up

```sh
docker rm lab1-train lab1-predict
docker rmi lab1:latest
docker volume rm lab1-output
rm -rf output lab1-image.tar.gz
```

Or run `./run_lab.sh --clean`, which does all of this except deleting `./output`.

## Dataset

Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository, bundled with scikit-learn as `sklearn.datasets.load_breast_cancer`. It has 569 samples (212 malignant, 357 benign) with 30 numeric features computed from images of cell nuclei. Nothing is downloaded at runtime.
