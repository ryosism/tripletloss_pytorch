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
from TcnDataset import TcnDataset

# pretrained models
from cnn_finetune import make_model

# parameters
import paramConfig as cfg

# matplotlib and tsne
import matplotlib.pyplot as plt
import visualizeTriplet

# logger
import logging
import time

# ================================================================== #

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imageRootDir', '-i', help='imageRootDir')
    parser.add_argument('--jsonDir', '-j', help='jsonDir')
    parser.add_argument('--frameLengthCSV', '-c', help='Path_to_frame_length.csv')

    return parser.parse_args()


def quintupletLoss(anchor_feat, po_1_feat, po_2_feat, ne_1_feat, ne_2_feat, alpla=cfg.LOSS_MARGIN):
    pdist = nn.PairwiseDistance(p=2, keepdim=True)

    da1p1 = pdist(anchor_feat, po_1_feat)
    da1n1 = pdist(anchor_feat, ne_1_feat)

    if cfg.ENABLE_SWAP:
        # swapの実装
        dp1n1 = pdist(po_1_feat, ne_1_feat)
        da1n1 = torch.min(dp1n1, da1n1)

        da1p2 = pdist(anchor_feat, po_2_feat)
        da1p1 = torch.max(da1p1, da1p2)

        da1n2 = pdist(anchor_feat, ne_2_feat)
        da1n1 = torch.min(da1n1, da1n2)
        # swapをすることによって一番lossが大きい組み合わせにする

    loss = torch.clamp(da1p1 - da1n1 + alpla, min=0.0)

    return loss


def tripletLoss(a_feat, po_feat, ne_feat, alpla=cfg.LOSS_MARGIN):
    pdist = nn.PairwiseDistance(p=2, keepdim=True)
    d_ap = pdist(a_feat, po_feat)
    d_an = pdist(a_feat, ne_feat)

    swap = pdist(po_feat, ne_feat)
    d_an = torch.min(d_an, swap)

    loss = torch.clamp(d_ap - d_an + alpla, min=0.0)

    return loss


def clusteredTripletLoss(anchor_feat, po_feat_list, ne_feat_list, alpla=cfg.LOSS_MARGIN):
    pdist = nn.PairwiseDistance(p=2, keepdim=True)

    anchorPositiveDistList = [pdist(anchor_feat, po_feat) for po_feat in po_feat_list]
    anchorNegativeDistList = [pdist(anchor_feat, ne_feat) for ne_feat in ne_feat_list]
    maxAnchorPositiveDist = max(anchorPositiveDistList)
    minAnchorNegativeDist = min(anchorNegativeDistList)

    loss = torch.clamp(maxAnchorPositiveDist - minAnchorNegativeDist + alpla, min=0.0)

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
    img_size = cfg.IMG_SIZE

    # logger
    logger = logging.getLogger('LoggingTest')
    logger.setLevel(20)

    fh = logging.FileHandler('./log/train_log_ex{}.log'.format(cfg.NUM_TRAINEX))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    logger.addHandler(sh)

    # dataloader
    tcnDataset = TcnDataset(imageRootDir=args.imageRootDir, jsonDir=args.jsonDir, frameLengthCSV=args.frameLengthCSV, anchorSize=img_size, numOfSamplesForSwap=int(cfg.NUM_OF_SAMPLES_FOR_SWAP))
    train_loader = torch.utils.data.DataLoader(dataset=tcnDataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # model
    model = make_model('inception_v3', num_classes=cfg.NET_DIMENTIONS, pretrained=True, input_size=img_size)
    device = torch.device('cuda')
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # loss function
    # triplet_loss = nn.TripletMarginLoss(margin=cfg.LOSS_MARGIN, p=2, swap=True)

    train_loss = []
    for epoch_idx in range(1, epochNum+1, 1):
        model.train()

        epochStartTime = time.time()

        interval_loss_sum = 0
        epoch_loss_sum = 0
        for batch_idx, (anchorTensor, positiveSampleTensorList, negativeSampleTensorList) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            # この処理によって、batch_negaはtorch.Size([numOfSamplesForSwap * batch_size, 3, height, width])になっている
            batch_posi = torch.empty([0, 3, img_size[0], img_size[1]])
            for positiveSampleTensor in positiveSampleTensorList:
                batch_posi = torch.cat((batch_posi, positiveSampleTensor), dim=0)

            batch_nega = torch.empty([0, 3, img_size[0], img_size[1]])
            for negativeSampleTensor in negativeSampleTensorList:
                batch_nega = torch.cat((batch_nega, negativeSampleTensor), dim=0)

            anchor_vecs_batch = model(anchorTensor.to(device))
            posi_vec_batch = model(batch_posi.to(device))
            nega_vec_batch = model(batch_nega.to(device))

            sample_posi_vec_batch = [torch.unsqueeze(vec, 0) for vec in posi_vec_batch]
            sample_nega_vec_batch = [torch.unsqueeze(vec, 0) for vec in nega_vec_batch]

            # batchsizeが4，numOfSamplesForSwapが3，anchorに対する入力クラスタをそれぞれa,b,c,dだとしたら[abcdabcdabcd]という感じにconcatされて積み上がってベクトルまでくる
            # 下の式によって[[a,a,a], [b,b,b], [c,c,c], [d,d,d]]という風に分割されてclusterdTripletLossに投げられる
            positive_vecs_list = [sample_posi_vec_batch[idx::batch_size] for idx in range(batch_size)]
            negative_vecs_list = [sample_nega_vec_batch[idx::batch_size] for idx in range(batch_size)]

            all_loss = 0
            for anchor_vec, positive_vecs, negative_vecs in zip(anchor_vecs_batch, positive_vecs_list, negative_vecs_list):
                if cfg.LOSS_FUNCTION == 'tripletloss':
                    loss = tripletLoss(a_feat=anchor_vec, po_feat=positive_vecs[0], ne_feat=negative_vecs[0])

                elif cfg.LOSS_FUNCTION == 'clusteredTripletLoss':
                    loss = clusteredTripletLoss(anchor_feat=anchor_vec, po_feat_list=positive_vecs, ne_feat_list=negative_vecs)

                else:
                    loss = quintupletLoss(
                        anchor_vec,
                        positive_vecs[0],
                        positive_vecs[1],
                        negative_vecs[0],
                        negative_vecs[1],
                    )

                    if cfg.TSNE_DEBUG:
                        fileName = "./ex{}/scatter_iter{}_idx{}.pdf".format(cfg.NUM_TRAINEX.zfill(2), str(epoch_idx).zfill(3), str(batch_idx).zfill(5))
                        if int(epoch_idx) > 0:
                            if loss > 10.0:
                                visualizeTriplet.featToTsne(
                                    featList=[
                                        anchor_vec.clone().detach().cpu().numpy().astype(np.float64),
                                        positive_vecs[0].clone().detach().cpu().numpy().astype(np.float64),
                                        positive_vecs[1].clone().detach().cpu().numpy().astype(np.float64),
                                        negative_vecs[0].clone().detach().cpu().numpy().astype(np.float64),
                                        negative_vecs[1].clone().detach().cpu().numpy().astype(np.float64)
                                    ],
                                    fileName=fileName,
                                    imgList=[
                                        torch.squeeze(anchorTensor[0]).permute(1,2,0).numpy(),
                                        torch.squeeze(positiveSampleTensorList[0][0]).permute(1,2,0).numpy(),
                                        torch.squeeze(positiveSampleTensorList[1][0]).permute(1,2,0).numpy(),
                                        torch.squeeze(negativeSampleTensorList[0][0]).permute(1,2,0).numpy(),
                                        torch.squeeze(negativeSampleTensorList[1][0]).permute(1,2,0).numpy()
                                    ]
                                )
                all_loss = all_loss + loss

            all_loss = all_loss/batch_size
            all_loss.backward()
            optimizer.step()

            interval_loss_sum = interval_loss_sum + all_loss.clone().detach().item()
            epoch_loss_sum = epoch_loss_sum + all_loss.clone().detach().item()

            if batch_idx * batch_size % log_interval == 0:
                logger.log(30, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss:{}'.format(
                    epoch_idx,
                    batch_idx * batch_size,
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    interval_loss_sum * batch_size / log_interval
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
        torch.save(model, './ex{}/model_ex{}_epoch{}.ckpt'.format(cfg.NUM_TRAINEX, cfg.NUM_TRAINEX, str(epoch_idx).zfill(3)))
        torch.save(model.state_dict(), './ex{}/params_ex{}_epoch{}.ckpt'.format(cfg.NUM_TRAINEX, cfg.NUM_TRAINEX, str(epoch_idx).zfill(3)))

        epochEndTime = time.time()
        logger.log(30, "Epoch Time : {}(sec)".format(int(epochEndTime - epochStartTime)))
        logger.log(30, "############################################################\n")


    import pdb; pdb.set_trace()


if __name__ == '__main__':
    args = parse_arguments()
    if not Path("./ex{}".format(str(cfg.NUM_TRAINEX).zfill(2))).exists():
        Path("./ex{}".format(str(cfg.NUM_TRAINEX).zfill(2))).mkdir(parents=True)
    train()
