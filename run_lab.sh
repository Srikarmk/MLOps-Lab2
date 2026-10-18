#!/usr/bin/env bash
# Runs the whole lab end to end: build -> save/load round trip -> train -> predict -> export.
#
#   ./run_lab.sh            run everything, keep the image, volume and containers afterwards
#   ./run_lab.sh --clean    run everything, then remove the containers, image, volume and tarball
#
# IMAGE and VOLUME can be overridden, e.g. IMAGE=mylab:v1 ./run_lab.sh

set -euo pipefail

IMAGE="${IMAGE:-lab1:latest}"
VOLUME="${VOLUME:-lab1-output}"
TRAIN_CONTAINER="lab1-train"
PREDICT_CONTAINER="lab1-predict"
TARBALL="lab1-image.tar.gz"
CLEAN=false
[ "${1:-}" = "--clean" ] && CLEAN=true

cd "$(dirname "$0")"

step() { printf '\n==> %s\n' "$*"; }
read_accuracy() {
    docker run --rm -v "$VOLUME:/app/output" --entrypoint cat "$IMAGE" /app/output/metrics.json \
        | grep '"accuracy"' | tr -dc '0-9.'
}

step "Checking Docker"
if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found. Install Docker Desktop: https://docs.docker.com/get-docker/" >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "Docker is installed but the daemon isn't running. Start Docker Desktop and try again." >&2
    exit 1
fi
docker version --format 'Client {{.Client.Version}}, Engine {{.Server.Version}}'

step "Removing containers left from a previous run"
docker rm -f "$TRAIN_CONTAINER" "$PREDICT_CONTAINER" >/dev/null 2>&1 || true

step "Building $IMAGE"
docker build -t "$IMAGE" .
docker image ls "$IMAGE" --format 'Image size: {{.Size}}'

step "Creating named volume $VOLUME"
docker volume create "$VOLUME" >/dev/null
docker volume inspect "$VOLUME" --format 'Volume data stored at: {{.Mountpoint}}'

step "Baseline training run (temporary container, removed on exit with --rm)"
docker run --rm -v "$VOLUME:/app/output" "$IMAGE" train | grep -E 'Test (accuracy|ROC-AUC)'
accuracy_before=$(read_accuracy)

step "Saving the image to $TARBALL"
id_before=$(docker image inspect "$IMAGE" --format '{{.Id}}')
docker save "$IMAGE" | gzip > "$TARBALL"
ls -lh "$TARBALL" | awk '{print "Tarball size: " $5}'

step "Removing $IMAGE locally and loading it back from $TARBALL"
docker rmi "$IMAGE" >/dev/null
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "$IMAGE is still present after docker rmi" >&2
    exit 1
fi
echo "$IMAGE removed"
gunzip -c "$TARBALL" | docker load
id_after=$(docker image inspect "$IMAGE" --format '{{.Id}}')

step "Training with the loaded image (container $TRAIN_CONTAINER, kept after it exits)"
docker run --name "$TRAIN_CONTAINER" -v "$VOLUME:/app/output" "$IMAGE" train
accuracy_after=$(read_accuracy)

step "Predicting (container $PREDICT_CONTAINER reads the model from the same volume)"
docker run --name "$PREDICT_CONTAINER" -v "$VOLUME:/app/output" -v "$(pwd)/samples:/app/input:ro" "$IMAGE" \
    predict /app/input/new_samples.csv

step "Checking the save/load round trip"
echo "Image ID before save: $id_before"
echo "Image ID after load:  $id_after"
echo "Accuracy before: $accuracy_before, after: $accuracy_after"
if [ "$id_before" = "$id_after" ] && [ "$accuracy_before" = "$accuracy_after" ]; then
    echo "Round trip OK: the loaded image is identical and reproduces the same result."
else
    echo "Round trip mismatch." >&2
    exit 1
fi

step "Copying the volume contents to ./output"
mkdir -p output
docker cp "$PREDICT_CONTAINER:/app/output/." output/
ls -l output

if $CLEAN; then
    step "Cleaning up"
    docker rm "$TRAIN_CONTAINER" "$PREDICT_CONTAINER" >/dev/null
    docker rmi "$IMAGE" >/dev/null
    docker volume rm "$VOLUME" >/dev/null
    rm -f "$TARBALL"
    echo "Removed containers, $IMAGE, volume $VOLUME and $TARBALL (./output is kept)"
else
    step "Containers kept for inspection"
    docker ps -a --filter "name=lab1-" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
    echo "View their output with: docker logs $TRAIN_CONTAINER   (or open them in Docker Desktop)"
fi

step "Done"
