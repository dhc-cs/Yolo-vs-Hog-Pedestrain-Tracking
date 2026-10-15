import sys
import pandas as pd
import numpy as np

# read both input files
predictions = pd.read_csv(sys.argv[1])
groundTruth = pd.read_csv(sys.argv[2])


totalFrames = 242
iouThreshold = 0.5
cols = ["X", "Y", "Width", "Height"]
tp, fp, fn = 0, 0, 0




for frame in range(totalFrames):
    # get info for all boxes in current frame and corresponding gt frame
    boxes = predictions.loc[predictions["Frame"] == frame, cols].to_numpy(dtype=float)
    gtBoxes = groundTruth.loc[groundTruth["Frame"] == frame, cols].to_numpy(dtype=float)


    # handle false negatives and false positives
    if len(boxes) == 0:
        fn += len(gtBoxes)
        continue
    if len(gtBoxes) == 0:
        fp += len(boxes)
        continue

    # top left corner of overlap
    topLeft = np.maximum(boxes[:, None, :2], gtBoxes[None, :, :2])
    
    # bottom right corner of overlap
    bottomRight = np.minimum(boxes[:, None, :2] + boxes[:, None, 2:], gtBoxes[None, :, :2] + gtBoxes[None, :, 2:])

    # height and width of overlap
    overlap = np.maximum(0, bottomRight - topLeft)

    # area of overlap
    area = overlap[:, :, 0] * overlap[:, :, 1]

    #aread of pedicted boxes
    predictedAreas = boxes[:, 2] * boxes[:, 3]

    # area of ground truth boxes
    gtAreas = gtBoxes[:, 2] * gtBoxes[:, 3]

    #union area 
    unioon = predictedAreas[:, None] + gtAreas[None, :] - area

    iou = area / unioon
    matches = 0; 

    #match pair with highest iou
    while iou.max() >= iouThreshold:
        predictedIdx, trueIdx = np.unravel_index(iou.argmax(), iou.shape)
        matches += 1

        #remove matches from iou matrix
        iou[predictedIdx, :] = 0
        iou[:, trueIdx] = 0

    # update counts
    tp += matches
    fp += len(boxes) - matches
    fn += len(gtBoxes) - matches

#calculate precision
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

#calculate recall
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

#calculate f1
f1 = 2* tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

if precision is None:
    print("Precision: N/A")
else:
    print(f"Precision: {precision:.4f}")

if recall is None:
    print("Recall: N/A")
else:
    print(f"Recall: {recall:.4f}")

if f1 is None:
    print("F1: N/A")
else:
    print(f"F1: {f1:.4f}")