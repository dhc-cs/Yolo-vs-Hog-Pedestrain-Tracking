# HOG and YOLO Pedestrian Detection and Tracking

## Overview

For this project, I am comparing traditional computer vision techniques with deep learning for pedestrian detection. I am using OpenCV’s HOG/SVM detector and a pretrained YOLOv5s model on the same Jetson Orin Nano.

My goal is to compare detection accuracy and processing performance using the same input video. Additionally, I plan to track pedestrians by assigning IDs and drawing trajectories using the center of each bounding box.

## Current Status

So far, both implementations can:

- Accept a video file through the command line.
- Detect pedestrians and draw bounding boxes.
- Display the processed video.
- Save an annotated output video.

HOG and YOLO run independently of one another. Tracking, accuracy testing, and comparable performance measurements have not yet been implemented.

## Requirements

The project has been tested on a Jetson Orin Nano using:

- OpenCV 4.5.4
- C++
- g++
- make
- pkg-config

YOLO runs inference through OpenCV DNN. Neither implementation currently uses CUDA or Pthreads.

## Build and Run

Run the following commands from the repository root.

### HOG

```bash
cd hog
make clean
make
./hog --video=/path/to/input.mp4 --bb --store=hog_output.mp4
```

The `--bb` option displays bounding boxes. The `--store` option saves the processed video.

HOG also supports `--show` to display a separate Sobel gradient visualization.

### YOLO

```bash
cd yolo
make clean
make
./yolo --video=/path/to/input.mp4 --store=yolo_output.mp4
```

Run YOLO from the `yolo` directory so it can find `config/yolov5s.onnx`.

Exit each program using `CTRL + C` . Saving the output video is optional.

## Models and Input Videos

HOG uses OpenCV’s pretrained SVM pedestrian detector. YOLO uses the pretrained YOLOv5s ONNX model. I did not train either model.

The YOLO model must be located at `yolo/config/yolov5s.onnx`. The starter code and models came from the [doleron YOLOv5 example](https://github.com/doleron/yolov5-opencv-cpp-python).

Input videos must be supplied separately. For accuracy testing, I will use the same unannotated video for both methods.

## Evaluation

This project is currently in the proof-of-concept phase. I have not yet determined which method is more accurate or faster.

I plan to measure:

- Precision and recall
- Processing frame rate
- Tracking accuracy and incorrect ID changes.

The saved video’s playback frame rate does not represent the detector’s processing speed.

## Next Steps

- Implement bounding-box center tracking.
- Assign pedestrian IDs and draw trajectories.
- Complete accuracy and performance testing.
- Tune detector settings without changing model weights.
- Investigate threading after detection and tracking work.

## Sources and Acknowledgments

HOG was adapted from [OpenCV’s pedestrian detection sample](https://github.com/opencv/opencv/blob/4.5.4/samples/cpp/peopledetect.cpp).

YOLO was adapted from the [doleron YOLOv5 OpenCV example](https://github.com/doleron/yolov5-opencv-cpp-python).

Video saving was based on [OpenCV’s video-writing tutorial](https://docs.opencv.org/4.5.4/d7/d9e/tutorial_video_write.html).

The Makefiles were adapted from [Professor Sam Siewert’s starter code](https://github.com/sbsiewertcsu/cv4_computer_vision_starter_code/blob/main/simpler-capture-4/Makefile). Professor Siewert also provided the Jetson and course materials.
