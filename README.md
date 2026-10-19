# HOG and YOLO Pedestrian Detection and Tracking

## Overview

I compare OpenCV’s HOG/SVM detector with pretrained YOLOv5n on a Jetson Orin Nano using the same video. Both programs detect pedestrians, assign tracking IDs, draw trajectories, and optionally save annotated videos and prediction CSVs. I did not train either model.


## Requirements

- OpenCV 4.5.4
- g++, make, and pkg-config
- Python 3 with NumPy, pandas, and motmetrics for evaluation

## Input Data

The evaluation uses these files:

```text
data/V016_8s.mp4
data/V016_8s_labels.csv
```

The video contains 242 frames from the [Caltech Pedestrians dataset](https://data.caltech.edu/records/f6rph-90m20).

I used MATLAB Online and Piotr Dollár’s toolbox to export the frames and labels, then FFmpeg to create the MP4. The export script is `data/export_V016.m`.

## Build and Run

Start from the repository root.

### HOG

```bash
cd hog
make
./hog --video=../data/V016_8s.mp4 --bb --log=output.csv --store=output.mp4
```

### YOLO

```bash
cd yolo
make
./yolo --video=../data/V016_8s.mp4 --log=output.csv --store=output.mp4
```

Run YOLO from its directory so it can find `config/yolov5n.onnx`.

- `--video`: input video
- `--log`: save predictions to CSV
- `--store`: save annotated video
- HOG `--bb`: draw bounding boxes
- HOG `--show`: display Sobel edges


Press **q** or **Escape** to exit. HOG also uses **Space** to switch Default/Daimler detectors. Let the full clip finish before evaluating.

## Evaluation


From the repository root, create the environment once:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas motmetrics
```

For later runs:

```bash
source .venv/bin/activate
bash evalAll.sh
```

The script evaluates `hog/output.csv` and `yolo/output.csv` against `data/V016_8s_labels.csv`.

- `evaluate.py`: precision, recall, and F1
- `evalTracking.py`: IDF1 and ID switches

Both evaluators use an IoU threshold of 0.50. The current evaluation is configured for 242 frames.


Each row is one detected box. Frames without detections have no rows.


## Sources and Acknowledgments

- HOG: [OpenCV pedestrian detection sample](https://github.com/opencv/opencv/blob/4.5.4/samples/cpp/peopledetect.cpp)
- YOLO: [doleron YOLOv5 example](https://github.com/doleron/yolov5-opencv-cpp-python)
- Video saving: [OpenCV video-writing tutorial](https://docs.opencv.org/4.5.4/d7/d9e/tutorial_video_write.html)
- Makefiles: [Professor Sam Siewert’s starter code](https://github.com/sbsiewertcsu/cv4_computer_vision_starter_code/blob/main/simpler-capture-4/Makefile)
- Data preparation: [Piotr Dollár’s MATLAB Toolbox](https://github.com/pdollar/toolbox) and FFmpeg
- Tracking evaluation: [motmetrics](https://github.com/cheind/py-motmetrics)
- Dataset reference: [Pedestrian Detection: A Benchmark](https://ieeexplore.ieee.org/document/5206631)
