#!/bin/sh
set -e

case "$1" in
    train)
        shift
        exec python main.py "$@"
        ;;
    predict)
        shift
        exec python predict.py "$@"
        ;;
    help|--help|-h)
        echo "Usage: docker run [docker options] <image> [train | predict [input.csv] | <command>]"
        echo "  train            train the model and write it to \$OUTPUT_DIR"
        echo "  predict          score random samples from the dataset with the saved model"
        echo "  predict FILE     score the rows of a CSV file (30 feature columns)"
        echo "  <command>        run any other command, e.g. 'sh' or 'python --version'"
        ;;
    *)
        exec "$@"
        ;;
esac
