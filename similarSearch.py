# ================================================================== #
# pytorch
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# argparse
import argparse

# pathfinder
from pathlib import Path

# Dataset
from similarSearchDataset import *

# pretrained models
from cnn_finetune import make_model

# parameters
import paramConfig as cfg

# matplotlib
import matplotlib.pyplot as plt

# logger
import logging

# Search Engine
import nmslib

# ================================================================== #

def parse_arguments():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


def loadImage(imagePath, device):
    img = cv2.imread(str(imagePath)).astype(np.float32)
    img = torch.from_numpy(img).permute(2,0,1)
    img = torch.unsqueeze(img, 0).to(device)

    return img


def test(logger, epochParamPath):
    # Load testdata
    testDirRoot = Path(cfg.TEST_DIR_ROOT)
    testDirList = testDirRoot.glob("00?")

    # Configure GPU
    device = torch.device('cuda')

    # Load trained model
    logger.log(30, "Loading trained model {}.".format(str(Path(epochParamPath).name)))
    model = make_model('inception_v4', num_classes=1000, pretrained=True, input_size=(768, 1024))
    paramPathList = Path("./{}")
    model.load_state_dict(torch.load(epochParamPath))
    model = model.to(device)
    model.eval()
    logger.log(30, "model {} was loaded.".format(str(Path(epochParamPath).name)))

    # Predict the closest frame in all frames
    for testDir in testDirList:

        # フレームディレクトリからDataloaderを作成
        similarSearchDataset = SimilarSearchDataset(videoFramePathDir=str(testDir/"frame30"))
        videoFrameDataLoader = torch.utils.data.DataLoader(
            dataset=similarSearchDataset,
            batch_size=cfg.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=6,
            pin_memory=True
        )

        # 手順画像は直接リストで作成
        recipeOrderImageList = [path for path in (testDir/"recipeOrder").glob("*.png")]

        # 動画フレームの画像を特徴抽出
        videoFrameFeatList = np.empty((0, 1000), int)
        for idx, batch in enumerate(videoFrameDataLoader, start=1):
            with torch.no_grad():
                feat = model(batch.to(device)).clone().detach().cpu().numpy()
            videoFrameFeatList = np.vstack([videoFrameFeatList, feat])
            if idx % 10 == 0:
                logger.log(30, "{} images are extracted.".format(str(idx * cfg.TEST_BATCH_SIZE)))

        logger.log(30, "{} images are extracted.".format(str(len(videoFrameFeatList))))

        # 手順画像も特徴抽出
        recipeOrderFeatList = []
        for idx, imagePath in enumerate(recipeOrderImageList, start=1):
            img = loadImage(imagePath, device)
            with torch.no_grad():
                feat = model(img).clone().detach().cpu().numpy()
            recipeOrderFeatList.append(feat)

        logger.log(30, "Extarcted all features.")

        # Init nmslib
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(videoFrameFeatList)
        index.createIndex({'post': 2})

        queryJson = open("query_{}.json".format(str(testDir.name)), "w")
        candidateJson = open("candidate_{}.json".format(str(testDir.name)), "w")

        QUERY = []
        CANDIDATE = []

        for recipeOrderFileName, recipeOrderFeat in zip(recipeOrderImageList, recipeOrderFeatList):
            ids, distances = index.knnQuery(recipeOrderFeat, k=5)
            logger.log(30, recipeOrderFileName)
            logger.log(30, ids)
            logger.log(30, distances)

            CANDIDATE_PER_ORDER = []
            for id in ids:
                CANDIDATE_PER_ORDER.append(str(testDir/"frame30") + str(int(id)*30).zfill(5)))

            CANDIDATE.append(CANDIDATE_PER_ORDER)



if __name__ == '__main__':
    # logger
    logger = logging.getLogger('LoggingTest')
    logger.setLevel(20)

    fh = logging.FileHandler('test_log_ex{}.log'.format(cfg.NUM_EX))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    logger.addHandler(sh)

    args = parse_arguments()
    test(logger=logger, epochParamPath=cfg.TRAINED_PARAM)
