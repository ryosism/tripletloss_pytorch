# ================================================================== #
# pytorch
import torch
import torchvision
# import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image

#argparse
import argparse

# dataLoader
from pathlib import Path
import json
import csv
import random

# ================================================================== #

# You should build your custom dataset as below.
class TcnDataset(torch.utils.data.Dataset):
    def __init__(self, imageRootDir, jsonDir, frameLengthCSV):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.imageRootDir = Path(imageRootDir)
        self.jsonList = Path(jsonDir).glob("*.json")
        self.jsonList = sorted([jsonPath for jsonPath in self.jsonList])

        self.anchorPathList = []
        self.positivePathList = []

        # 各動画の長さを取得
        with open(frameLengthCSV, "r") as f:
            self.CSV = csv.reader(f, delimiter=",", doublequote=True)
            self.CSV = [row for row in self.CSV]

        # jsonごとに
        for jsonPath in self.jsonList:
            with open(jsonPath) as f:
                JSON = json.load(f)
                for recipe in JSON["data"]:
                    pairs = recipe[1]["pic"]

                    # sampling for anchor
                    self.anchorPathList += [pair["anchor"] for pair in pairs]

                    # sampling for positive
                    self.positivePathList += [pair["positive"] for pair in pairs]


    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        anchor_path = self.anchorPathList[index]
        anchor = self.transform(Image.open(anchor_path))

        positive_1_Path = self.positivePathList[index]
        positive_1 = self.transform(Image.open(positive_1_Path))

        positiveIndexNum = int(Path(positive_1_Path).stem)

        while(1):
            # positiveから50フレームまでの距離も実質positiveに
            positiveRandIndexNum = random.choice(range(positiveIndexNum - 50, positiveIndexNum + 50, 10))
            positiveAroundPath = Path(str(Path(positive_1_Path).parent)+"/"+str(positiveRandIndexNum).zfill(5)+".png")
            if Path(positiveAroundPath).exists():
                positive_2_Path = positiveAroundPath
                break

        positive_2 = self.transform(Image.open(positive_2_Path))

        # negativeの取得 ############
        # 動画のフレームの長さを取得
        videoNum = int((Path(positive_1_Path).parent.parent).name)
        frameLength = int(self.CSV[videoNum-1][1])

        # positiveから総フレーム数/4以上離れているフレームをnegativeに
        # notNegativeRangeMin↓              notNegativeRangeMax↓
        #      |nnnnnnnnnnnnn|--------------p|p|p--------------|nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn|
        notNegativeRangeMin = max(0, positiveIndexNum - int(frameLength/4))
        notNegativeRangeMax = min(frameLength, positiveIndexNum + int(frameLength/4))
        negativeRange = [*range(0,frameLength)[:notNegativeRangeMin], *range(0,frameLength)[notNegativeRangeMax:]]

        while(1):
            negativeIndexNum = random.choice(negativeRange)
            negativePath = Path(str(Path(positive_1_Path).parent)+"/"+str(negativeIndexNum).zfill(5)+".png")
            if Path(negativePath).exists():
                negative_1_Path = str(negativePath)
                break

        while(1):
            negativeIndexNum = random.choice(negativeRange)
            negativePath = Path(str(Path(positive_1_Path).parent)+"/"+str(negativeIndexNum).zfill(5)+".png")
            if Path(negativePath).exists():
                negative_2_Path = str(negativePath)
                break

        negative_1 = self.transform(Image.open(negative_1_Path))
        negative_2 = self.transform(Image.open(negative_2_Path))

        # anchor2も作ってData Augmentationはあり

        # return self.transform(anchor), self.transform(positive_1), self.transform(positive_2), self.transform(negative_1), self.transform(negative_2)
        return anchor, positive_1, positive_2, negative_1, negative_2

    def __len__(self):
        return len(self.anchorPathList)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageRootDir', '-i', help='imageRootDir')
    parser.add_argument('--jsonDir', '-j', help='jsonDir')
    parser.add_argument('--frameLengthCSV', '-c', help='Path_to_frame_length.csv')

    return parser.parse_args()


def main():
    args = parse_arguments()
    tcnDataset = TcnDataset(imageRootDir=args.imageRootDir, jsonDir=args.jsonDir, frameLengthCSV=args.frameLengthCSV)

    train_loader = torch.utils.data.DataLoader(dataset=tcnDataset,batch_size=1,shuffle=False)
    train_loader = iter(train_loader)

    for batch_idx, (anchor, positive_1, positive_2, negative_1, negative_2) in enumerate(train_loader):
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
