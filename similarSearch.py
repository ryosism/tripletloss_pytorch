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
import os

# Dataset
from similarSearchDataset import SimilarSearchDataset
import cv2

# pretrained models
from cnn_finetune import make_model

# parameters
import paramConfig as cfg

# logger
import logging

# matplotlib and tsne
import matplotlib.pyplot as plt
import visualizeTriplet

# Search Engine
import nmslib

# json writer
import json

# ================================================================== #

def loadImage(imagePath, device):
    img = cv2.imread(str(imagePath)).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = torch.unsqueeze(img, 0).to(device)

    return img


def top5(top5Dist, top5Index, newDist, newIndex, newFileName, top5FileName):
    for i, dist in enumerate(top5Dist):
        if float(newDist) < dist:
            for index in top5Index:
                if newIndex in range(index - 30, index + 30):
                    return top5Dist, top5Index, top5FileName

            top5Dist.insert(i, float(newDist))
            top5Dist.pop(5)
            top5Index.insert(i, newIndex)
            top5Index.pop(5)
            top5FileName.insert(i, str(os.path.abspath(newFileName)))
            top5FileName.pop(5)

            return top5Dist, top5Index, top5FileName

    return top5Dist, top5Index, top5FileName


def test_singleParam_singleVideo(logger, model, paramPath, frameDir, orderDir):
    # Configure GPU
    device = torch.device('cuda')

    # Load trained model
    logger.log(30, "Loading trained model {}.".format(str(paramPath.name)))
    model.load_state_dict(torch.load(str(paramPath)))
    model = model.to(device)
    model.eval()
    logger.log(30, "model {} was loaded.".format(str(paramPath.name)))

    # フレームディレクトリからDataloaderを作成
    similarSearchDataset = SimilarSearchDataset(videoFramePathDir=frameDir)
    videoFrameDataLoader = torch.utils.data.DataLoader(
        dataset=similarSearchDataset,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )
    # t-sneに投げる用の画像(numpy)のリストを作成
    imgListToTsne = []
    featShapeToTsne = []

    # 手順画像は直接リストで作成
    recipeOrderImageList = [path for path in Path(orderDir).glob("*.png")]

    # 動画フレームの画像を特徴抽出
    # TODO: メモリに全ての特徴ベクトルをスタックさせるため、数十GBのメモリを食ってしまっている
    #       バッチごとに全ての手順画像とそれぞれ距離比較して、都度ランキングを更新して行った方がメモリ的に安全
    videoFrameFeatList = np.empty((0, int(cfg.TEST_NET_DIMENTIONS)), int)
    for idx, batch in enumerate(videoFrameDataLoader, start=1):
        batchArray = np.array(batch.cpu())
        imgListToTsne += [img for img in batchArray]
        with torch.no_grad():
            feat = model(batch.to(device)).clone().detach().cpu().numpy()
        videoFrameFeatList = np.vstack([videoFrameFeatList, feat])
        if idx % 10 == 0:
            logger.log(30, "{} images are extracted.".format(str(idx * cfg.TEST_BATCH_SIZE)))

    logger.log(30, "{} images are extracted.".format(str(len(videoFrameFeatList))))
    featShapeToTsne = videoFrameFeatList

    # 手順画像も特徴抽出
    recipeOrderFeatList = []
    for idx, imagePath in enumerate(recipeOrderImageList, start=1):
        img = loadImage(imagePath, device)
        imgArray = np.array(img.cpu())
        imgListToTsne += [imgArray.reshape(imgArray.shape[1:])]
        with torch.no_grad():
            feat = model(img).clone().detach().cpu().numpy()
        featShapeToTsne = np.vstack([featShapeToTsne, feat])
        recipeOrderFeatList.append(feat)

    logger.log(30, "Extarcted all features.")

    # 手順画像ごとに距離が近い動画フレーム画像を検索
    query = []
    candidate = []

    # # Init nmslib
    # index = nmslib.init(method='hnsw', space='l2')
    # index.addDataPointBatch(videoFrameFeatList)
    # index.createIndex({'post': 2})
    #
    # for recipeOrderFileName, recipeOrderFeat in zip(recipeOrderImageList, recipeOrderFeatList):
    #     ids, distances = index.knnQuery(recipeOrderFeat, k=5)
    #     logger.log(30, recipeOrderFileName)
    #     logger.log(30, ids)
    #     logger.log(30, distances)
    #
    #     CANDIDATE_PER_ORDER = []
    #     for id in ids:
    #         CANDIDATE_PER_ORDER.append(str(Path(frameDir)/"frame30") + str(int(id)*30).zfill(5) + ".png")
    #
    #     QUERY.append(recipeOrderFileName)
    #     CANDIDATE.append(CANDIDATE_PER_ORDER)
    #
    visualizeTriplet.featToTsne(
        featList = featShapeToTsne,
        fileName = "./ex{}/scatter_image_video_{}_p{}".format(str(cfg.NUM_TESTEX).zfill(2), str(Path(args.testDirPath).name), str(cfg.TEST_PERPLEXITY)),
        imgList = [np.transpose(img, (1, 2, 0)) for img in imgListToTsne],
        perplexity = float(cfg.TEST_PERPLEXITY)
    )

    for recipeOrderFileName, recipeOrderFeat in zip(recipeOrderImageList, recipeOrderFeatList):
        logger.log(30, recipeOrderFileName)

        top5Dist = [100000000, 100000000, 100000000, 100000000, 100000000]
        top5Index = [-1, -1, -1, -1, -1]
        top5FileName = ["", "", "", "", ""]

        for idx, videoFrameFeat in enumerate(videoFrameFeatList, start=1):
            dist = np.linalg.norm(recipeOrderFeat - videoFrameFeat)
            videoFrameFileName = str(Path(frameDir)) + "/" + str(int(idx) * 30).zfill(5) + ".png"
            top5Dist, top5Index, top5FileName = top5(
                top5Dist=top5Dist,
                top5Index=top5Index,
                newDist=dist,
                newIndex=idx,
                newFileName=videoFrameFileName,
                top5FileName=top5FileName
            )

        for fileName, dist in zip(top5FileName, top5Dist):
            logger.log(30, "【{}】 {}".format(dist, fileName))

        query.append(str(recipeOrderFileName))
        candidate.append(top5FileName)

        #     if dist < minDist:
        #         minIdx = idx
        #         minDist = dist
        #
        # minDistVideoFrameName = str(Path(frameDir)/"frame30") + str(int(minIdx)*30).zfill(5) + ".png"
        # minDistVideoFrameDist = minDist

    return query, candidate


def test_singleParam_allVideos(logger, epochParamPath):
    # Init JSON object
    JSON = []

    # Load testdata
    testDirRoot = Path(cfg.TEST_DIR_ROOT)
    testDirList = testDirRoot.glob("00?")

    model = make_model('inception_v3', num_classes=cfg.TEST_NET_DIMENTIONS, pretrained=True, input_size=cfg.TEST_IMG_SIZE)

    # Predict the closest frame in all frames
    for testDir in testDirList:
        frameDir = str(testDir / "frame30")
        orderDir = str(testDir / "orderImages")
        QUERY, CANDIDATE = test_singleParam_singleVideo(
            logger=logger, model=model, paramPath=epochParamPath, frameDir=frameDir, orderDir=orderDir)
        JSON.append(CANDIDATE)

    return QUERY, JSON


def test_allParams_singleVideo(logger, paramDirPath, testDirPath):
    # Init JSON object
    JSON = []

    model = make_model('inception_v3', num_classes=cfg.TEST_NET_DIMENTIONS, pretrained=True, input_size=cfg.TEST_IMG_SIZE)

    # Predict the closest frame in all frames
    paramPathList = sorted(Path(paramDirPath).glob("params*"))

    for paramPath in paramPathList:
        frameDir = str(testDirPath / "frame30")
        orderDir = str(testDirPath / "orderImages")
        QUERY, CANDIDATE = test_singleParam_singleVideo(
            logger=logger, model=model, paramPath=paramPath, frameDir=frameDir, orderDir=orderDir)
        JSON.append(CANDIDATE)

    return QUERY, JSON


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paramDirPath', '-p', help='Path to ParamDirPath')
    parser.add_argument('--testDirPath', '-t', help='Path to testDirPath')

    return parser.parse_args()


if __name__ == '__main__':
    if Path(cfg.LOGFILE).exists():
        Path(cfg.LOGFILE).unlink()

    # logger
    logger = logging.getLogger('LoggingTest')
    logger.setLevel(20)
    fh = logging.FileHandler(cfg.LOGFILE)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    logger.addHandler(sh)

    args = parse_arguments()

    QUERY, CANDIDATE = test_allParams_singleVideo(logger=logger, paramDirPath=Path(args.paramDirPath), testDirPath=Path(args.testDirPath))

    with open("./ex{}/query_{}.json".format(str(cfg.NUM_TESTEX).zfill(2), str(Path(args.testDirPath).name)), "w") as f:
        json.dump(QUERY, f, indent=4, ensure_ascii=False, separators=(',', ': '))

    with open("./ex{}/candidate_{}.json".format(str(cfg.NUM_TESTEX).zfill(2), str(Path(args.testDirPath).name)), "w") as f:
        json.dump(CANDIDATE, f, indent=4, ensure_ascii=False, separators=(',', ': '))
