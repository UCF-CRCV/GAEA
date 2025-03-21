import csv
import json
import os
from tqdm import tqdm
import numpy as np
import argparse
import uuid

from geopy.distance import distance

parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", type=str, help='Path to the model predictions')
parser.add_argument("--gt_path", type=str, help='Path to GT')
args = parser.parse_args()

gt_path = args.gt_path
pred_path = args.pred_path


print(gt_path)
print(pred_path)

def distance_accuracy(targets, preds, distance_thresholds, faulty):
    total = len(targets)
    correct = {str(dis):0 for dis in distance_thresholds}
    gd_list = []

    no_classifications = []
    for i in tqdm(targets):
        if i not in faulty:
            # print(i)

            try:
                gd = distance(preds[i], targets[i]).km
                gd_list.append(gd)
                for dis in distance_thresholds:
                    if gd <= dis:
                        correct[str(dis)] += 1
            except:
                no_classifications.append(i)
                continue
    acc = {k:v/total for k,v in correct.items()}

    gd_avg = sum(gd_list)/total
    return acc, gd_avg, no_classifications

targets = {}

yfcc4k = False
if "yfcc4k" in gt_path:
    la, lo = 3, 4
    yfcc4k = True
elif "gws15k" in gt_path:
    la, lo = 1, 2
else:
    la, lo = 2, 3

extension = '.jpg' if yfcc4k else ''
with open(gt_path, 'r') as f:
    csvreader = csv.reader(f)
    next(csvreader, None)
    for line in csvreader:
        img_target = line[0] + extension
        targets[img_target] = [float(line[la]), float(line[lo])]


if os.path.exists(pred_path):
    preds = json.load(open(pred_path))
    pred_set = set([k for k in preds.keys()])
else:
    preds = {}
    pred_set = set()

faulty = {}

distance_thresholds = [2500, 750, 200, 25, 1] # km
accuracy_results = {}
error_results = np.Inf

acc, avg_distance_error, no_classifications = distance_accuracy(targets, preds, distance_thresholds, faulty.keys())
print(f"There was an error in calculating {len(no_classifications)} files.")
for dis in distance_thresholds:
    print(f"Accuracy at {dis} km: {acc[str(dis)]}")
    accuracy_results[f'acc_{dis}_km'] = acc[str(dis)]
print(f"Average distance error: {avg_distance_error} km")

