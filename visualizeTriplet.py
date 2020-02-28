# ================================================================== #
# pytorch
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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

# matplotlib and tsne
import matplotlib.pyplot as plt
from bhtsne import tsne
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

# logger
import logging

# draw rectangle
from PIL import Image, ImageDraw

# ================================================================== #

def scatter_image(filename, Y, imgList):
    """
    Args:
        feature_x: x座標
        feature_y: y座標
        imgList: 画像(numpy)のリスト
    """
    feature_x = Y[:, 0]
    feature_y = Y[:, 1]
    print("plot : [{}]".format(Y[:100, :]))
    print("plot : [{}]".format(Y[-5:, :]))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    xlim = [feature_x.min()-20, feature_x.max()+20]
    ylim = [feature_y.min()-20, feature_y.max()+20]

    # color = ['black', 'red', 'red', 'blue', 'blue'] #anchor:黒 positive:赤 negative:青

    print("{} images were recieved.".format(len(imgList)))
    print("{} features were tsned.".format(len(feature_x)))

    feature_x = [x for x in feature_x]
    feature_y = [y for y in feature_y]

    mabikiFeatureX = feature_x[:-15][::10] + feature_x[-15:]
    mabikiFeatureY = feature_y[:-15][::10] + feature_y[-15:]
    mabikiImgList = imgList[:-15][::10] + imgList[-15:]

    for idx, (x, y, img) in enumerate(zip(mabikiFeatureX, mabikiFeatureY, mabikiImgList)):
        # import pdb; pdb.set_trace()
        #
        # # 枠線を描く
        # height, width, channel = img.shape
        # img = (img*256).astype('uint8') # PIL対応の配列にする
        # img = Image.fromarray(img)
        # canvas = ImageDraw.Draw(img)
        # canvas.rectangle([(0, 0), (width, height)], fill=None, outline=color[idx], width=5)
        #

        img = np.asarray(img).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, ch = img.shape
        aspectRatio = img_width/img_height

        bb = Bbox.from_bounds(x, y, 4 * aspectRatio, 4)
        bb2 = TransformedBbox(bb, ax.transData)
        bbox_image = BboxImage(bb2, norm=None, origin=None, clip_on=False)

        bbox_image.set_data(img)
        ax.add_artist(bbox_image)

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    plt.savefig(filename, dpi=600)
    plt.close()


def featToTsne(featList, fileName, imgList, perplexity):
    featList = np.array(featList)
    Y = tsne(featList, perplexity=perplexity)
    scatter_image(filename=fileName, Y=Y, imgList=imgList)
