import pandas as pd 
import sys
import motmetrics as mm

predictions =pd.read_csv(sys.argv[1])
gt =pd.read_csv(sys.argv[2])
totalFrames=int(sys.argv[3])

iouThreshold = 0.5
cols = ["X","Y","Width","Height"]

#check for dup track ids in same frame
if predictions.duplicated(["Frame", "TrackID"]).any():
    raise SystemExit("Error: the same tracking id appears multiple times in the same frame.")



results = mm.MOTAccumulator(auto_id=True)

for frame in range(totalFrames):

    #get current frame predictions and ground truth
    currentPredict = predictions.loc[predictions["Frame"] == frame]
    currentGT = gt.loc[gt["Frame"] == frame]

    #get current frame predicted and ground trueth IDs
    predictIDs = currentPredict["TrackID"].to_numpy(dtype=int)
    gtIDs = currentGT["ID"].to_numpy(dtype=int)

    #get current frame predicted and ground trueth bounding boxes
    predictBoxes = currentPredict[cols].to_numpy(dtype=float)
    gtBoxes = currentGT[cols].to_numpy(dtype=float)



    #compute IoU matrix
    distanceMatrix = mm.distances.iou_matrix(gtBoxes, predictBoxes, max_iou=1 - iouThreshold)

    results.update(gtIDs, predictIDs, distanceMatrix)

# compute metrics
summary = mm.metrics.create().compute(results, metrics=["idf1", "num_switches"], name="result")
idf1 = summary.loc["result", "idf1"]
switches = int(summary.loc["result", "num_switches"])

if pd.isna(idf1):
    print("tracking IDF1: n/a")
else:
    print(f"tracking IDF1: {idf1:.2%}")

print(f"tracking switches: {switches}")