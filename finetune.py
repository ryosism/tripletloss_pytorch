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

# ================================================================== #


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imageRootDir', '-i', help='imageRootDir')
    parser.add_argument('--jsonDir', '-j', help='jsonDir')
    parser.add_argument('--frameLengthCSV', '-c', help='Path_to_frame_length.csv')

    return parser.parse_args()


def quintupletLoss(anchor, po_1_feat, po_2_feat, ne_1_feat, ne_2_feat, alpla=0.5):
    da1p1 = anchor.data.norm() - po_1_feat.data.norm()
    da1n1 = anchor.data.norm() - ne_1_feat.data.norm()
    dp1p2 = po_1_feat.data.norm() - po_2_feat.data.norm()

    loss = max(da1p1 - da1n1 + dp1p2 - alpla, 0)

    return loss


def trapletLoss(a_feat, po_feat, ne_feat, alpla=0.5):
    d_ap = a_feat.data.norm() - po_feat.data.norm()
    d_an = a_feat.data.norm() - ne_feat.data.norm()

    loss = max(d_ap - d_an + alpla, 0)

    return loss


def plot_graph(train_loss, outputPath):
    if not Path(outputPath).parent.exists():
        Path(outputPath).parent.mkdir(parents=True)

    fig = plt.figure(figsize=(16, 8))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(["train_loss"], loc="upper right")

    plt.plot(range(len(train_loss)), train_loss, "r", linewidth=1.5, linestyle="-")
    plt.savefig(outputPath, bbox_inches="tight")
    plt.close()


def train():
    # parameters
    epochNum = cfg.EPOCH
    log_interval = cfg.LOG_INTERVAL
    batch_size = cfg.BATCH_SIZE

    # logger
    logger = logging.getLogger('LoggingTest')
    logger.setLevel(20)

    fh = logging.FileHandler('train_log_ex{}.log'.format(cfg.NUM_EX))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    logger.addHandler(sh)

    # dataloader
    tcnDataset = TcnDataset(imageRootDir=args.imageRootDir, jsonDir=args.jsonDir, frameLengthCSV=args.frameLengthCSV)
    train_loader = torch.utils.data.DataLoader(dataset=tcnDataset, batch_size=batch_size, shuffle=True, num_workers=6)
    # train_loader = iter(train_loader)

    # model
    model = make_model('inception_v4', num_classes=1000, pretrained=True, input_size=(384, 384))
    device = torch.device('cuda')
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # loss function
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    train_loss = []
    for epoch_idx in range(1, epochNum+1, 1):
        model.train()
        interval_loss_sum = 0
        epoch_loss_sum = 0
        for batch_idx, (anchor, positive_1, positive_2, negative_1, negative_2) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            sample = [anchor, positive_1, positive_2, negative_1, negative_2]
            sample = [sample.to(device) for sample in sample]
            sample_vec = [model(img) for img in sample]

            anchor, positive_1, positive_2, negative_1, negative_2 = sample_vec

            loss = triplet_loss(anchor, positive_1, negative_1)
            # loss = quintupletLoss(anchor, po_1_feat, po_2_feat, ne_1_feat, ne_2_feat)
            loss.backward()
            optimizer.step()

            interval_loss_sum = interval_loss_sum + loss.clone().detach()
            epoch_loss_sum = epoch_loss_sum + loss.clone().detach()

            if batch_idx % log_interval == 0:
                logger.log(30, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss:{}'.format(
                    epoch_idx,
                    batch_idx * batch_size,
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    interval_loss_sum/log_interval
                ))
                interval_loss_sum = 0

        epoch_loss = epoch_loss_sum/batch_idx

        logger.log(30, "#################### Epoch {} summary ######################".format(epoch_idx))
        logger.log(30, "epoch_loss : {}".format(epoch_loss))
        train_loss.append(epoch_loss)
        plot_graph(train_loss, cfg.GRAPH_PDF)

        # if this epoch's loss is minimum, save the models!
        if not len(train_loss) == 1:
            if epoch_loss < min(train_loss[:-1]):
                logger.log(30, "Minimum loss was {}(epoch{}) -> {}(epoch{})".format(
                    min(train_loss),
                    train_loss.index(min(train_loss)),
                    epoch_loss,
                    epoch_idx
                ))
                torch.save(model, './ex{}/model_ex{}_epoch{}.ckpt'.format(cfg.NUM_EX, cfg.NUM_EX, str(epoch_idx)))
                torch.save(model.state_dict(), './ex{}/params_ex{}_epoch{}.ckpt'.format(cfg.NUM_EX, cfg.NUM_EX, str(epoch_idx)))

        logger.log(30, "############################################################\n")


    import pdb; pdb.set_trace()


if __name__ == '__main__':
    args = parse_arguments()
    train()
