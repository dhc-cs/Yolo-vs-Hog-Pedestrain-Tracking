#ifndef COM_TRACKER_HPP
#define COM_TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

class CenterTracker {
private:
    // A record of a tracked object.
    struct TrackingRecord {
        int id;
        Point center; // most recent COM
        int consecutiveMisses; // num of frames id has not been detected
        vector<Point> centerHistory; // previous COMs
    };

    vector<TrackingRecord> multipleRecords;
    int nextId = 1; // for assigning subsequent IDs to new pedestrians
    double maxDistance;
    int maxMissedFrames;

public:
    CenterTracker(double distance = 80.0, int missed = 5) {
        maxDistance = distance;
        maxMissedFrames = missed;
    }

    // tracking logic
    void update(const vector<Rect>& frameBBs) {
        vector<Point> centers;

        // Calculate centers using the Assignment 3 calculation.
        for (int i = 0; i < frameBBs.size(); ++i) {
            const Rect& boundingBox = frameBBs[i];

            if (boundingBox.width <= 0 || boundingBox.height <= 0)
                continue;

            int xbar = boundingBox.x + (boundingBox.width - 1) / 2;
            int ybar = boundingBox.y + (boundingBox.height - 1) / 2;

            // add calculated COM to centers vector
            centers.push_back(Point(xbar, ybar));
        }

        // flags to keep track of used centers
        vector<bool> centersUsed(centers.size(), false);

        for (size_t i = 0; i < multipleRecords.size(); i++) {
            TrackingRecord& currentTrackingRecord = multipleRecords[i];
            int closest = -1;
            double shortestDistance = maxDistance;

            // Find the nearest unused center within the distance limit.
            for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
                if (centersUsed[i]) // skip if center is already used
                    continue;

                // calculate euclidean distance current and detected COMs
                double dx = double(centers[i].x) - currentTrackingRecord.center.x;
                double dy = double(centers[i].y) - currentTrackingRecord.center.y;
                double distance = sqrt(dx * dx + dy * dy);

                //greedy neearst neighbor approach to find closest COM
                if (distance <= shortestDistance) {
                    shortestDistance = distance;
                    closest = i;
                }
            }

            // update existing record is closest found, else increment consecutiveMisses
            if (closest != -1) {
                currentTrackingRecord.center = centers[closest]; //update COM
                currentTrackingRecord.consecutiveMisses = 0; // reset consecutiveMisses
                currentTrackingRecord.centerHistory.push_back(currentTrackingRecord.center); // add new COM to history
                centersUsed[closest] = true; // mark center as used

                // Limit the trajectory to the latest 100 observations.
                if (currentTrackingRecord.centerHistory.size() > 100) {
                    currentTrackingRecord.centerHistory.erase(currentTrackingRecord.centerHistory.begin());
                }
            } else {
                currentTrackingRecord.consecutiveMisses++;
            }
        }

        // first time detection of new COMs, add to multipleRecords
        for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
            if (centersUsed[i]) // skip if center is already used
                continue;

            TrackingRecord newTrack;
            newTrack.id = nextId++;
            newTrack.center = centers[i];
            newTrack.consecutiveMisses = 0;
            newTrack.centerHistory.push_back(centers[i]);
            multipleRecords.push_back(newTrack);
        }

        // Remove tracks missing for too many frames.
        for (size_t i = 0; i < multipleRecords.size();) {
            if (multipleRecords[i].consecutiveMisses > maxMissedFrames) {
                multipleRecords.erase(multipleRecords.begin() + i);
            } else {
                ++i;
            }
        }
    }

    // visualization logic 
    void draw(Mat& frame) const {
        Scalar color(255, 165, 0);

        for (size_t i = 0; i < multipleRecords.size(); i++) {
            const TrackingRecord& currentTrackingRecord = multipleRecords[i];
            if (currentTrackingRecord.consecutiveMisses != 0)
                continue;

            // Connect successive centers
            for (size_t i = 1; i < currentTrackingRecord.centerHistory.size(); ++i) {
                line(frame,
                         currentTrackingRecord.centerHistory[i - 1],
                         currentTrackingRecord.centerHistory[i],
                         color, 2);
            }

            circle(frame, currentTrackingRecord.center, 4, color, FILLED);

            putText(frame,
                        "ID " + to_string(currentTrackingRecord.id),
                        currentTrackingRecord.center + Point(6, -6),
                        FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2);
        }
    }
};

#endif