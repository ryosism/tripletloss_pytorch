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
    def __init__(self, imageRootDir, jsonDir, frameLengthCSV, anchorSize, numOfSamplesForSwap):
        self.transform = transforms.Compose([
            transforms.Resize(anchorSize),
            transforms.ToTensor()
        ])
        self.imageRootDir = Path(imageRootDir)
        self.anchorSize = anchorSize
        self.jsonList = Path(jsonDir).glob("*.json")
        self.jsonList = sorted([jsonPath for jsonPath in self.jsonList])
        self.numOfSamplesForSwap = int(numOfSamplesForSwap)

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
        anchorTensor = self.transform(Image.open(anchor_path))


        # positiveの取得 ##################################################################
        positiveSamplePathList = []
        positiveSampleTensorList = []

        annotatedPositivePath = self.positivePathList[index]
        positiveSamplePathList.append(annotatedPositivePath)
        annotatedPositiveIndexNum = int(Path(annotatedPositivePath).stem)

        if not self.numOfSamplesForSwap == 1:
            while(1):
                # positiveから50フレームまでの距離も実質positiveに
                # TODO: ここ50フレームってことは1.6秒とかじゃないか？もう少し長くしてもいいかもしれない....
                # TODO: このままだとnumOfSamplesForSwapが5以上の場合、ここで無限ループする

                # FIXED: positiveから10秒前10秒後までsemiPositiveにした、この場合だとnumOfSamplesForSwapは14まで許容される
                # 00:00 |-----------------------|pppppppppppppp|P| 40:00 ←Positiveが一番末尾にあった場合10秒後はフレームがないため無限ループを起こす

                positiveRandIndexNum = random.choice(range(annotatedPositiveIndexNum - 300, annotatedPositiveIndexNum + 300, 10))
                positiveAroundPath = Path(str(Path(annotatedPositivePath).parent)+"/"+str(positiveRandIndexNum).zfill(7)+".png")
                if Path(positiveAroundPath).exists():
                    if positiveAroundPath in positiveSamplePathList: #もうすでにリストに存在しているAroundPathは無視する
                        continue

                    positiveSamplePathList.append(positiveAroundPath)
                    if len(positiveSamplePathList) == self.numOfSamplesForSwap:
                        break

        # positiveSamplePathListの中の画像を全て読み込む
        positiveSampleTensorList = [self.transform(Image.open(str(path))) for path in positiveSamplePathList]


        # negativeの取得 ################################################################
        negativeSamplePathList = []
        negativeSampleTensorList = []

        # 動画のフレームの長さを取得
        videoNum = int((Path(annotatedPositivePath).parent.parent).name)
        frameLength = int(self.CSV[videoNum-1][1])

        # positiveから総フレーム数/4以上離れているフレームをnegativeRangeに
        # ##################################イメージ図####################################################
        # notNegativeRangeMin↓                                 ↓notNegativeRangeMax
        #      |nnnnnnnnnnnnn|--------------p|p|p--------------|nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn|
        #     　       ↑negativeRange                                            ↑negativeRange
        # ##############################################################################################
        notNegativeRangeMin = max(0, annotatedPositiveIndexNum - int(frameLength/4))
        notNegativeRangeMax = min(frameLength, annotatedPositiveIndexNum + int(frameLength/4))
        negativeRange = [*range(0,frameLength)[:notNegativeRangeMin], *range(0,frameLength)[notNegativeRangeMax:]]

        while(1):
            negativeIndexNum = random.choice(negativeRange)
            negativePath = Path(str(Path(annotatedPositivePath).parent)+"/"+str(negativeIndexNum).zfill(7)+".png")
            if Path(negativePath).exists():
                if negativePath in negativeSamplePathList:
                    continue

                negativeSamplePathList.append(negativePath)
                if len(negativeSamplePathList) == self.numOfSamplesForSwap:
                    break

        # negativeSamplePathListの中の画像を全て読み込む
        negativeSampleTensorList = [self.transform(Image.open(str(path))) for path in negativeSamplePathList]

        return anchorTensor, positiveSampleTensorList, negativeSampleTensorList

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
    tcnDataset = TcnDataset(imageRootDir=args.imageRootDir, jsonDir=args.jsonDir, frameLengthCSV=args.frameLengthCSV, anchorSize=(384, 512), numOfSamplesForSwap=3)

    train_loader = torch.utils.data.DataLoader(dataset=tcnDataset,batch_size=4,shuffle=False, num_workers=8)
    train_loader = iter(train_loader)

    for batch_idx, (anchorTensor, positiveSampleTensorList, negativeSampleTensorList) in enumerate(train_loader):
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
