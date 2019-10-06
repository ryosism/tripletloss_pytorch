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
from TcnDataset import *

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


def test(logger):
    # Load testdata
    testDirRoot = Path(cfg.TEST_DIR_ROOT)
    testDirList = testDirRoot.glob("00?")

    # Configure GPU
    device = torch.device('cuda')

    # Load trained model
    logger.log(30, "Loading trained model {}.".format(str(Path(cfg.TRAINED_PARAM).name)))
    model = make_model('inception_v4', num_classes=1000, pretrained=True, input_size=(384, 384))
    model.load_state_dict(torch.load(cfg.TRAINED_PARAM))
    model = model.to(device)
    logger.log(30, "model {} was loaded.".format(str(Path(cfg.TRAINED_PARAM).name)))



    # Predict the closest frame in all frames
    for testDir in testDirList:
        import pdb; pdb.set_trace()

        recipeOrderImageList = [path for path in (testDir/"recipeOrder").glob("*.png")]
        videoFrameList = [path for path in (testDir/"frame30").glob("*.png")]

        ##########################################################################
        # 動画フレームの画像サイズを検出
        videoHeight, videoWidth, channel = cv2.imread(str(videoFrameList[0])).shape

        # 動画フレームの画像を特徴抽出
        videoFeatList = np.empty((0, 1000), int)
        batch = torch.empty(0, channel, videoHeight, videoWidth).to(device)

        # バッチサイズ分のパスのリストを取得
        for idx, imagePathBatch in enumerate([videoFrameList[i:i+cfg.TEST_BATCH_SIZE] for i in range(0,len(videoFrameList),cfg.TEST_BATCH_SIZE)], start=1):

        for idx, imagePath in enumerate(videoFrameList, start=1):
            img = loadImage(imagePath, device)
            batch = torch.cat((batch, img), dim=0)

            # バッチサイズ分揃ったら、モデルに投げる、投げたら結合する
            if idx % cfg.TEST_BATCH_SIZE == 0:
                feat = model(batch).clone().detach().cpu().numpy()
                videoFeatList = np.vstack([videoFeatList, feat])

                batch = torch.empty(0, channel, videoHeight, videoWidth).to(device)

            if idx % cfg.TEST_BATCH_SIZE*10 == 0:
                logger.log(30, "{}".format(imagePath))
                logger.log(30, "{} images are extracted.".format(str(idx)))

        # バッチに収まらなかった残りの分も投げる
        feat = model(batch).clone().detach().cpu().numpy()
        videoFeatList = np.vstack([videoFeatList, feat])
        logger.log(30, "{} images are extracted.".format(str(idx)))

        # 手順画像も特徴抽出
        recipeOrderFeatList = []
        for idx, imagePath in enumerate(recipeOrderImageList, start=1):
            img = loadImage(imagePath, device)
            feat = model(img).clone().detach().cpu().numpy()
            recipeOrderFeatList.append(feat)

        ##########################################################################

        logger.log(30, "Extarcted all features.")

        # Init nmslib
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(videoFeatList)
        index.createIndex({'post': 2}, print_progress=True)

        for fileName, recipeOrderFeat in zip(recipeOrderImageList, recipeOrderFeatList):
            ids, distances = index.knnQuery(recipeOrderFeat, k=5)
            logger.log(30, fileName)
            logger.log(30, ids)
            logger.log(30, distances)

        # 溜まったキャッシュを削除、これしないとOutOfMemoryになる
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # logger
    logger = logging.getLogger('LoggingTest')
    logger.setLevel(20)

    fh = logging.FileHandler('test_log_ex{}.log'.format(cfg.NUM_EX))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    logger.addHandler(sh)

    args = parse_arguments()
    test(logger=logger)
