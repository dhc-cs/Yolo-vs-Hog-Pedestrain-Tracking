#!/usr/bin/env bash
set -e

echo "========== HOG =========="
python evaluate.py hog/hogOutput.csv data/V016_8s_labels.csv
python evalTracking.py hog/hogOutput.csv data/V016_8s_labels.csv 242

echo
echo "========== YOLO =========="
python evaluate.py yolo/yoloOutput.csv data/V016_8s_labels.csv
python evalTracking.py yolo/yoloOutput.csv data/V016_8s_labels.csv 242