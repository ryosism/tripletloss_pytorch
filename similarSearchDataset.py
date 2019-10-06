# ================================================================== #
# pytorch
import torch
# import torchvision
# import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import cv2
cv2.setNumThreads(0)

#argparse
import argparse

# dataLoader
from pathlib import Path
import json
import csv
import random

# ================================================================== #

# You should build your custom dataset as below.
class SimilarSearchDataset(torch.utils.data.Dataset):
    def __init__(self, videoFramePathDir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.imageRootDir = Path(videoFramePathDir)
        self.videoFrameList = sorted(self.imageRootDir.glob("*.png"))


    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        img = cv2.imread(str(self.videoFrameList[index])).astype(np.float32)
        img = torch.from_numpy(img).permute(2,0,1)

        return img


    def __len__(self):
        return len(self.videoFrameList)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videoFramePathDir', '-v', help='videoFramePathDir')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    similarSearchDataset = SimilarSearchDataset(args.videoFramePathDir)
    import pdb; pdb.set_trace()
