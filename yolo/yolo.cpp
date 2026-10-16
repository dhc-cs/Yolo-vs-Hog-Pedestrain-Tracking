#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "../comTracker.hpp"
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::dnn;


static const string keys =
    "{ video v | | use video as input }"
    "{ store   | | store the output video }"
    "{ log     | | save detections to a csv file }"; 

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
    Net net = readNetFromONNX("config/yolov5n.onnx");

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    const int input_size = 640;
    const float confidence_threshold = 0.25f;
    const float nms_threshold = 0.40f;

    Mat frame;
    CenterTracker tracker(80.0, 5); // maxDistance = 80.0, maxMissedFrames = 5

    // open csv file for logging detections 
    string filename = parser.get<string>("log"); // read the file named passed in the CL
    ofstream logFile; // output file stream object to write to a file 

    if(!filename.empty()) { // only log when file name is provided (litle sanity check)
        // create/open file for writing 
        logFile.open(filename); 

        // write the header 
        logFile << "Frame,TrackID,X,Y,Width,Height\n";
    }
    int frameCont = 0; // to match ground truth frame numbers

    double inferTime = 0.0; // to keep track of inference time

    // Video Loop 
    while (capture.read(frame)) {

        int64 t= getTickCount(); // start timer

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
        
        int64 inferStart = getTickCount(); // start inference timer
        net.forward(outputs, net.getUnconnectedOutLayersNames()); // run inference
        inferTime += (getTickCount() - inferStart) / getTickFrequency(); // end inference timer


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

        // non max supression
        vector<int> indices;
        NMSBoxes(
            boxes,
            scores,
            confidence_threshold,
            nms_threshold,
            indices);

        t = getTickCount() - t; // end timer

        // bounding box loop
        vector<Rect> finalBoxes;
        for (int idx : indices) {
            const Rect& box = boxes[idx];
            if (box.width <= 0 || box.height <= 0)
                continue;
            
            finalBoxes.push_back(box);



            rectangle(frame, box, Scalar(0, 255, 0), 2);
        }

        //update tracker and get ids
        vector<int> trackIDs = tracker.update(finalBoxes); 
        if (logFile.is_open()) {
            for(size_t i=0; i<finalBoxes.size(); ++i) {
                const Rect& box = finalBoxes[i];
                logFile << frameCont << "," << trackIDs[i] << "," << box.x << "," << box.y << "," << box.width << "," << box.height << "\n";
            }
        }

        tracker.draw(frame); // draw trajectories and IDs

        putText(frame, format("FPS: %.1f", getTickFrequency() / (double)t), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);

        imshow("YOLO Pedestrian Detection", frame);

        if (outputVideo.isOpened()) {
            outputVideo.write(frame); 
        }

        frameCont++; // increment frame counter for logging
        int key = waitKey(1);

        if (key == 27 || key == 'q')
            break;
    }

    cout << "Number of frames that were processed: " << frameCont << endl;

    if (frameCont > 0) {
        cout << "avg inference time: " << 1000.0 * inferTime / frameCont << " ms/frame" << endl;
    }

    destroyAllWindows();
    return 0;
}