// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <iomanip>
#include <syslog.h> // for logging to syslog
using namespace cv;
using namespace std;

#include <fstream>

#include "../comTracker.hpp"

class Detector
{
    enum Mode { Default, Daimler } m;
    HOGDescriptor hog, hog_d;
public:
    Detector() : m(Default), hog(), hog_d(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9)
    {
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        hog_d.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
    }
    void toggleMode() { m = (m == Default ? Daimler : Default); }
    string modeName() const { return (m == Default ? "Default" : "Daimler"); }
    vector<Rect> detect(InputArray img)
    {
        // Run the detector with default parameters. to get a higher hit-rate
        // (and more false alarms, respectively), decrease the hitThreshold and
        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        vector<Rect> found;
        if (m == Default)
            hog.detectMultiScale(img, found, 0, Size(8,8), Size(), 1.05, 2, false);
        else if (m == Daimler)
            hog_d.detectMultiScale(img, found, 0, Size(8,8), Size(), 1.05, 2, true);
        return found;
    }
    void adjustRect(Rect & r) const
    {
        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
    }
};
static const string keys = "{ help h   |   | print help message }"
                           "{ camera c | 0 | capture video from camera (device index starting from 0) }"
                           "{ video v  |   | use video as input }"
                           "{ bb       |   | draw bounding boxes }"
                           "{ show     |   | show the intermediate results }"
                           "{ store    |   | save the output video to a file }"
                           "{log       |   | save detections to csv file }";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates the use of the HoG descriptor.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int camera = parser.get<int>("camera");
    string file = parser.get<string>("video");

    // check bounding box flag
    bool bb = parser.has("bb");

    // check show flag
    bool show = parser.has("show");

    // retrieve the store flag
    string store = parser.get<string>("store");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    VideoCapture cap;
    if (file.empty())
        cap.open(camera);
    else
    {
        file = samples::findFileOrKeep(file);
        cap.open(file);
    }
    if (!cap.isOpened())
    {
        cout << "Can not open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << endl;
        return 2;
    }
    cout << "Press 'q' or <ESC> to quit." << endl;
    cout << "Press <space> to toggle between Default and Daimler detector" << endl;

    namedWindow("People detector", WINDOW_NORMAL);
    resizeWindow("People detector", 640, 480);

    if (show) {
        namedWindow("Sobel Edge Detection", WINDOW_NORMAL);
        resizeWindow("Sobel Edge Detection", 640, 480);
    }

    Detector detector;
    CenterTracker tracker(80.0, 5); // maxDistance = 80.0, maxMissedFrames = 5
    Mat frame;
    unsigned int frameCount = 0; // to keep track of frames 

    // open csv file 
    string fileName = parser.get<string>("log");
    ofstream logFile;

    if(!fileName.empty()) { // only log when file name is provided (litle sanity check)
        // create/open file for writing 
        logFile.open(fileName); 

        // write the header 
        logFile << "Frame,X,Y,Width,Height\n";
    }


    // create VideoWriter obj to save output video if store flag is set
    VideoWriter outputVideo;

    // adapted from the opencv tutorial on creating videos: https://docs.opencv.org/4.5.4/d7/d9e/tutorial_video_write.html
    if (parser.has("store")) {

        // get the codec code of the input video
        int ex = static_cast<int>(cap.get(CAP_PROP_FOURCC));

        // get the size of the input video
        Size S = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));

        // open the output video
        outputVideo.open(store, ex, cap.get(CAP_PROP_FPS), S, true);

        if (!outputVideo.isOpened()) {
            cout << "Could not open the output video for write: " << store << endl;
            return -1;
        }
    }

    for (;;)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }
        int64 t = getTickCount();
        vector<Rect> found = detector.detect(frame);
        t = getTickCount() - t;

        frameCount++; // increment frame count
        // log frame count and detections 
        // adapted from hw 2 edge_viewer.cpp
        syslog(LOG_INFO, "frameCount = %u detections = %zu", frameCount, found.size());


        // adapted from sobel.cpp from assignemnt 2
        if (show) {
            Mat src_gray, grad, grad_x, grad_y, abs_grad_x, abs_grad_y;

            int ddepth = CV_16S;
            int ksize = 3;
            int scale = 1;
            int delta = 0;

            // convert the frame to grayscale
            cvtColor(frame, src_gray, COLOR_BGR2GRAY);

            // calculating the x and y derivatives
            Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
            Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

            // converting back to CV_8U
            convertScaleAbs(grad_x, abs_grad_x);
            convertScaleAbs(grad_y, abs_grad_y);

            addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

            imshow("Sobel Edge Detection", grad);
        }


        // show the window
        {
            ostringstream buf;
            buf << "Mode: " << detector.modeName() << " ||| "
                << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
            putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        }

        unsigned int pedCount = 0; // to keep track of number of peds

        // bounding box loop
        for (vector<Rect>::iterator i = found.begin(); i != found.end(); ++i)
        {
            Rect &r = *i;
            detector.adjustRect(r);

            //
            if (logFile.is_open()) {
                logFile << (frameCount -1) << "," << r.x << "," << r.y << "," << r.width << "," << r.height << "\n";
            }

            pedCount++; // increment ped count

            // log the number of peds detected and the coordainates of each detected ped
            syslog(LOG_INFO, "frameCount = %u pedCount = %u x = %d y = %d width = %d height = %d", frameCount, pedCount, r.x, r.y, r.width, r.height);


            // draw bb if flag is set
            if (bb) { rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);}
        }

        // update tracker for current frame
        tracker.update(found);
        tracker.draw(frame);

        // write to output video if store flag is set
        if (outputVideo.isOpened()) {
            outputVideo.write(frame);
        }

        imshow("People detector", frame);
        // interact with user
        const char key = (char)waitKey(1);
        if (key == 27 || key == 'q') // ESC
        {
            cout << "Exit requested" << endl;
            break;
        }
        else if (key == ' ')
        {
            detector.toggleMode();
        }
    }

    cout << "The number of frames that were procesed was " << frameCount << endl;
    return 0;
}