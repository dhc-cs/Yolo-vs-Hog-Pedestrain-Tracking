// this program was adapted from contours.cpp, tracker.cpp, and my crosshairs.cpp program from assignment 1

#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::Mat f, mask; //f holds each pgm frame and mask is the threshold version
    int threshold = 97;

    // bounding rectangle and contours to visually track target 
    cv::Rect boundingR;
    std::vector<std::vector<cv::Point>> contours;

    int xbar,ybar; //to hold coordinates of COM
    unsigned int frameCnt = 0;

    //process each frame
    for (frameCnt=0; frameCnt<1260; frameCnt++) {

        // read/store current frame, save in mat object
        f= cv::imread(cv::format("../part5/frames/%04d.pgm", frameCnt), cv::IMREAD_GRAYSCALE);

        // apply open cv binary threshold to f and store in mask
        // pixels in laser are set to 255
        cv::threshold(f, mask, threshold, 255, cv::THRESH_BINARY);  

        // detecting contours. RETR_EXTERNAL specifies only the outline
        // CHAIN_APPROX_SIMPLE keeps only the points necessary to completely describe the outline of the shape, not everysingle point
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));


        if(!contours.empty()) {

            //calling built in opencv function to calculate surrounding rectangle 
            boundingR = cv::boundingRect(contours[0]);

            //calculate COM
            ybar = boundingR.y + (boundingR.height -1) /2;
            xbar = boundingR.x +(boundingR.width -1) /2;

            //crosshairs
            cv::line(f, cv::Point(xbar, 0), cv::Point(xbar, f.rows-1), cv::Scalar(255), 1);
            cv::line(f, cv::Point(0, ybar), cv::Point(f.cols-1, ybar), cv::Scalar(255), 1);

            //save frame as pgm image
            cv::imwrite(cv::format("tdFrames/%04d.pgm", frameCnt), f);
                
        }
    }
    return 0; 
}