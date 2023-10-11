import torch
import numpy as np
from tqdm import tqdm

def evaluate_voxel_prediction(prediction, gt):
    """  The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0"""

    with torch.no_grad():
        intersection = torch.sum(torch.logical_and(prediction,gt))
        union = torch.sum(torch.logical_or(prediction,gt))
        IoU = intersection / (union+1e-5)
        return IoU
    
def calc_average_iou(model, test_dataloader):
    model.eval()
    running_iou = []

    for inputs, targets, _, _ in tqdm(test_dataloader):
        inputs = inputs
        targets = targets

        outputs = model(inputs)
        iou = evaluate_voxel_prediction(outputs > 0.5, targets)
        running_iou.append(iou.item())

    average_iou = np.mean(running_iou)
    return average_iou

def interpolate_embs(emb1, emb2, n_steps = 10):
    ratios = torch.linspace(0, 1, steps=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * emb1 + ratio * emb2
        vectors.append(v)
    return vectors