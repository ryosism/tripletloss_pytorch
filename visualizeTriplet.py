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

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    xlim = [feature_x.min()-150, feature_x.max()+150]
    ylim = [feature_y.min()-150, feature_y.max()+150]

    color = ['black', 'red', 'red', 'blue', 'blue'] #anchor:黒 positive:赤 negative:青

    for idx, (x, y, img) in enumerate(zip(feature_x, feature_y, imgList)):
        import pdb; pdb.set_trace()

        # 枠線を描く
        height, width, channel = img.shape
        img = (img*256).astype('uint8') # PIL対応の配列にする
        img = Image.fromarray(img)
        canvas = ImageDraw.Draw(img)
        canvas.rectangle([(0, 0), (width, height)], fill=None, outline=color[idx], width=5)

        import pdb; pdb.set_trace()

        img = (np.asarray(img)/256).astype('float64')

        bb = Bbox.from_bounds(x, y, 100, 100)
        bb2 = TransformedBbox(bb, ax.transData)
        bbox_image = BboxImage(bb2, norm=None, origin=None, clip_on=False)

        bbox_image.set_data(img)
        ax.add_artist(bbox_image)

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    plt.savefig(filename, dpi=1200)
    plt.close()


def featToTsne(featList, fileName, imgList):
    featList = np.array(featList)
    featList = featList[0:featList.shape[0], 0, 0:featList.shape[2]]
    Y = tsne(featList, perplexity=0.5)
    scatter_image(filename=fileName, Y=Y, imgList=imgList)
