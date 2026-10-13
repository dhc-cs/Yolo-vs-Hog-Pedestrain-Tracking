#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "../comTracker.hpp"

using namespace cv;
using namespace std;
using namespace cv::dnn;


static const string keys =
    "{ video v | | use video as input }"
    "{ store   | | store the output video }";

int main(int argc, char **argv) {
    CommandLineParser parser(argc, argv, keys);
    string file = parser.get<string>("video");

    // open the video
    VideoCapture capture(file);

    // writing the output video
    // based on the official example: https://docs.opencv.org/4.5.4/d7/d9e/tutorial_video_write.html
    if (!capture.isOpened()) {
        cout << "Could not open the input video: " << file << endl;
        return -1;
    }

    string store = parser.get<string>("store"); // check store flag
    VideoWriter outputVideo; 

    if (parser.has("store")) {
        int ex = static_cast<int>(capture.get(CAP_PROP_FOURCC)); // get Codec Type- Int form

        Size S = Size((int) capture.get(CAP_PROP_FRAME_WIDTH), // Aquire input size
                    (int) capture.get(CAP_PROP_FRAME_HEIGHT));
        
        // Open the outpout video
        outputVideo.open(store, ex, capture.get(CAP_PROP_FPS), S, true);

        // check if video writer is opened
        if (!outputVideo.isOpened()) {
            cout << "Could not open the output video for write: " << store << endl;
            return -1;
        }
    }

    // load model
    Net net = readNetFromONNX("config/yolov5s.onnx");

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    const int input_size = 640;
    const float confidence_threshold = 0.25f;
    const float nms_threshold = 0.40f;

    Mat frame;
    CenterTracker tracker(80.0, 5); // maxDistance = 80.0, maxMissedFrames = 5

    while (capture.read(frame)) {
        int side = max(frame.cols, frame.rows);

        Mat padded = Mat::zeros(side, side, frame.type());

        frame.copyTo(padded(Rect(0, 0, frame.cols, frame.rows)));

        Mat blob = blobFromImage(
            padded,
            1.0 / 255.0,
            cv::Size(input_size, input_size),
            cv::Scalar(),
            true,
            false);

        net.setInput(blob); // load frame from yolo

        vector<Mat> outputs; // to hold output from yolo
        net.forward(outputs, net.getUnconnectedOutLayersNames()); // get output from yolo

        // reshape output. one row per prediction 
        Mat predictions = outputs[0].reshape(1, outputs[0].size[1]);

        vector<Rect> boxes;
        vector<float> scores;

        float scale = static_cast<float>(side) / input_size;

        //get detections
        for (int i = 0; i < predictions.rows; ++i) {
            const float *data = predictions.ptr<float>(i);

            float score = data[4] * data[5]; //calcualet confidence score

            if (score < confidence_threshold)
                continue;

            // only keep predictions were person scores highest 
            if (max_element(data + 5, data + 85) != data + 5)
                continue;

            // scaling out bb coordinates 
            int left = cvRound((data[0] - data[2] / 2) * scale);
            int top = cvRound((data[1] - data[3] / 2) * scale);
            int width = cvRound(data[2] * scale);
            int height = cvRound(data[3] * scale);

            Rect box(left, top, width, height); // create bb box

            // save current detectoin 
            boxes.push_back(box);
            scores.push_back(score);
        }

        // non max supression. removes overlapping boxes
        vector<int> indices;
        NMSBoxes(
            boxes,
            scores,
            confidence_threshold,
            nms_threshold,
            indices);

        //draw boxes
        vector<Rect> finalBoxes;

        for (int idx : indices) {
            const Rect& box = boxes[idx];
            if (box.width <= 0 || box.height <= 0)
                continue;
            
            finalBoxes.push_back(box);
            rectangle(frame, box, Scalar(0, 255, 0), 2);
        }

        //update tracker
        tracker.update(finalBoxes);
        tracker.draw(frame); // draw trajectories and IDs

        imshow("YOLO Pedestrian Detection", frame);

        if (outputVideo.isOpened()) {
            outputVideo.write(frame); 
        }

        int key = waitKey(1);

        if (key == 27 || key == 'q')
            break;
    }

    destroyAllWindows();
    return 0;
}