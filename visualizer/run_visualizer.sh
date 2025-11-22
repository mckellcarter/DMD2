#!/bin/bash
# Quick launch script for DMD2 visualizer

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if embeddings exist
EMBEDDINGS_DIR="data/embeddings"
MODEL="${1:-imagenet}"

if [ -d "$EMBEDDINGS_DIR" ] && [ "$(ls -A $EMBEDDINGS_DIR/*.csv 2>/dev/null)" ]; then
    # Find most recent embeddings file
    LATEST_EMBEDDINGS=$(ls -t $EMBEDDINGS_DIR/${MODEL}_umap_*.csv 2>/dev/null | head -n1)

    if [ -n "$LATEST_EMBEDDINGS" ]; then
        echo "Found embeddings: $LATEST_EMBEDDINGS"
        echo "Launching visualizer..."
        python visualization_app.py --embeddings "$LATEST_EMBEDDINGS" "${@:2}"
    else
        echo "No embeddings found for model: $MODEL"
        echo "Launching with dynamic UMAP mode..."
        python visualization_app.py --data_dir data "${@:2}"
    fi
else
    echo "No embeddings directory found."
    echo "Launching with dynamic UMAP mode..."
    python visualization_app.py --data_dir data "${@:2}"
fi
