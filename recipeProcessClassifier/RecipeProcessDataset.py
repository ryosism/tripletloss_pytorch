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

# testProcessing
from cnn_finetune import make_model

# ================================================================== #

# You should build your custom dataset as below.
class RecipeProcessDataset(torch.utils.data.Dataset):
    def __init__(self, imageRootDir, frameLengthCSV, anchorSize, trainOrTest, randomSeed):
        print("Creating dataset...")
        self.transform = transforms.Compose([
            transforms.Resize(anchorSize),
            transforms.ToTensor()
        ])
        self.imageRootDir = Path(imageRootDir)
        self.anchorSize = anchorSize

        # 各動画の長さを取得
        with open(frameLengthCSV, "r") as f:
            self.CSV = csv.reader(f, delimiter=",", doublequote=True)
            self.CSV = [row for row in self.CSV]
            self.videoLengthList = [row[1] for row in self.CSV]

        # 全動画のフレームリストを作成、教師の番号もここで生成
        self.allPathWithLabelList = []

        videoIDList = [video[0] for video in self.CSV]
        lengthList = [video[1] for video in self.CSV]

        for videoID, length in zip(videoIDList, lengthList):
            frameDirPath = self.imageRootDir / Path(str(videoID)) / Path("frame30")
            frameList = sorted(frameDirPath.glob("*.png"))

            frameSplitedList = [list(array) for array in np.array_split(frameList, 3)]
            for idx, frameSplited in enumerate(frameSplitedList):
                frameSplitedList[idx] = [idx] * len(frameSplited)

            videpProcessList = [e for inner_list in frameSplitedList for e in inner_list]
            pathWithLabelList = [[A, B] for A,B in zip(frameList, videpProcessList)]

            self.allPathWithLabelList += pathWithLabelList

        # trainとevalに分ける、該当する方のリストを残す
        numberOfTestSamples = 100
        random.seed(randomSeed)
        self.testSamples = random.sample(self.allPathWithLabelList, numberOfTestSamples)
        self.trainSamples = [e for e in self.allPathWithLabelList if not e in self.testSamples]

        if trainOrTest == "train":
            self.devidedDataset = self.trainSamples
        else:
            self.devidedDataset = self.testSamples

        print("Created dataset")


    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        imgPath = self.allPathWithLabelList[index][0]
        process = self.allPathWithLabelList[index][1]

        img = self.transform(Image.open(imgPath))

        return img, process


    def __len__(self):
        return len(self.allPathWithLabelList)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageRootDir', '-i', help='imageRootDir')
    parser.add_argument('--frameLengthCSV', '-c', help='Path_to_frame_length.csv')

    return parser.parse_args()


def main():
    args = parse_arguments()
    tcnDataset = RecipeProcessDataset(imageRootDir=args.imageRootDir, frameLengthCSV=args.frameLengthCSV, anchorSize=(384, 512), trainOrTest="Test", randomSeed=1)

    train_loader = torch.utils.data.DataLoader(dataset=tcnDataset,batch_size=4,shuffle=True, num_workers=8)
    train_loader = iter(train_loader)

    model = make_model('inception_v3', num_classes=8, pretrained=True, input_size=(384, 512))
    device = torch.device('cuda:3')

    set = next(train_loader)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
