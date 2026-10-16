#!/usr/bin/env bash
set -e

echo "========== HOG =========="
python evaluate.py hog/output.csv data/V016_8s_labels.csv
python evalTracking.py hog/output.csv data/V016_8s_labels.csv 242

echo
echo "========== YOLO =========="
python evaluate.py yolo/output.csv data/V016_8s_labels.csv
python evalTracking.py yolo/output.csv data/V016_8s_labels.csv 242
