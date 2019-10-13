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
import visualizeTriplet

# logger
import logging

# ================================================================== #

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imageRootDir', '-i', help='imageRootDir')
    parser.add_argument('--jsonDir', '-j', help='jsonDir')
    parser.add_argument('--frameLengthCSV', '-c', help='Path_to_frame_length.csv')
    parser.add_argument('--lossfunc', '-l', help="lossfunction ('tripletloss' or 'quintupletLoss')")

    return parser.parse_args()


def quintupletLoss(anchor_feat, po_1_feat, po_2_feat, ne_1_feat, ne_2_feat, numOfEx, numOfEpoch, numOfIdx, imgList, alpla=1.0):
    pdist = nn.PairwiseDistance(p=2, keepdim=True)

    da1p1 = pdist(anchor_feat, po_1_feat)
    da1n1 = pdist(anchor_feat, ne_1_feat)

    swap = pdist(po_1_feat, ne_1_feat)
    da1n1 = torch.min(swap, da1n1)

    dp1p2 = pdist(po_1_feat, po_2_feat)

    loss = torch.clamp(da1p1 - da1n1 + dp1p2 + alpla, min=0.0)

    fileName = "./ex{}/scatter_iter{}_idx{}.pdf".format(numOfEx.zfill(2), str(numOfEpoch).zfill(3), str(numOfIdx).zfill(5))

    if int(numOfEpoch) > 10:
        if loss > 10.0:
            visualizeTriplet.featToTsne(
                featList=[
                    anchor_feat.clone().detach().cpu().numpy().astype(np.float64),
                    po_1_feat.clone().detach().cpu().numpy().astype(np.float64),
                    po_2_feat.clone().detach().cpu().numpy().astype(np.float64),
                    ne_1_feat.clone().detach().cpu().numpy().astype(np.float64),
                    ne_2_feat.clone().detach().cpu().numpy().astype(np.float64)
                ],
                fileName=fileName,
                imgList=imgList
            )
            with open(str(Path(fileName).stem) + "txt") as f:
                f.write("da1p1 = {}".format(da1p1))
                f.write("da1n1 = {}, swap = {}".format(pdist(anchor_feat, ne_1_feat), swap))
                f.write("dp1p2 = {}".format(dp1p2))
                f.write("loss = {}".format(loss))

    return loss


def tripletLoss(a_feat, po_feat, ne_feat, alpla=1.0):
    pdist = nn.PairwiseDistance(p=2, keepdim=True)
    d_ap = pdist(a_feat, po_feat)
    d_an = pdist(a_feat, ne_feat)

    swap = pdist(po_feat, ne_feat)
    d_an = torch.min(d_an, swap)

    loss = torch.clamp(d_ap - d_an + alpla, min=0.0)

    return loss


def plot_graph(train_loss, outputPath):
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
    train_loader = torch.utils.data.DataLoader(dataset=tcnDataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # model
    model = make_model('inception_v4', num_classes=1000, pretrained=True, input_size=(768, 1024))
    device = torch.device('cuda')
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # loss function
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, swap=True)

    train_loss = []
    for epoch_idx in range(1, epochNum+1, 1):
        model.train()
        interval_loss_sum = 0
        epoch_loss_sum = 0
        for batch_idx, (anchor, positive_1, positive_2, negative_1, negative_2) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            batch_negaposi = positive_1
            batch_negaposi = torch.cat((batch_negaposi, positive_2), dim=0)
            batch_negaposi = torch.cat((batch_negaposi, negative_1), dim=0)
            batch_negaposi = torch.cat((batch_negaposi, negative_2), dim=0)

            anchor_vec = model(anchor.to(device))
            negaposi_vec = model(batch_negaposi.to(device))

            sample_vec = [torch.unsqueeze(vec, 0) for vec in negaposi_vec]
            positive_1_vec, positive_2_vec, negative_1_vec, negative_2_vec = sample_vec

            if args.lossfunc == 'tripletloss':
                # loss = triplet_loss(anchor_vec, positive_1_vec, negative_1_vec)
                loss = tripletLoss(a_feat=anchor_vec, po_feat=positive_1_vec, ne_feat=negative_1_vec)
            else:
                loss = quintupletLoss(
                    anchor_vec,
                    positive_1_vec,
                    positive_2_vec,
                    negative_1_vec,
                    negative_2_vec,
                    numOfEx=cfg.NUM_EX,
                    numOfEpoch=epoch_idx,
                    numOfIdx=batch_idx,
                    imgList=[
                        torch.squeeze(anchor).numpy(),
                        torch.squeeze(positive_1).numpy(),
                        torch.squeeze(positive_2).permute(2,1,0).numpy(),
                        torch.squeeze(negative_1).numpy(),
                        torch.squeeze(negative_2).permute(2,1,0).numpy()
                    ]
                )

            loss.backward()
            optimizer.step()

            interval_loss_sum = interval_loss_sum + loss.clone().detach().item()
            epoch_loss_sum = epoch_loss_sum + loss.clone().detach().item()

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
                    min(train_loss[:-1]),
                    train_loss.index(min(train_loss[:-1])),
                    epoch_loss,
                    epoch_idx
                ))
        torch.save(model, './ex{}/model_ex{}_epoch{}.ckpt'.format(cfg.NUM_EX, cfg.NUM_EX, str(epoch_idx)))
        torch.save(model.state_dict(), './ex{}/params_ex{}_epoch{}.ckpt'.format(cfg.NUM_EX, cfg.NUM_EX, str(epoch_idx)))

        logger.log(30, "############################################################\n")


    import pdb; pdb.set_trace()


if __name__ == '__main__':
    args = parse_arguments()
    if not Path("./ex{}".format(str(cfg.NUM_EX).zfill(2))).exists():
        Path("./ex{}".format(str(cfg.NUM_EX).zfill(2))).mkdir(parents=True)
    train()
