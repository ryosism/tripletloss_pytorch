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
from RecipeProcessDataset import RecipeProcessDataset

# pretrained models
from cnn_finetune import make_model

# parameters
import RecipeProcessParamConfig as cfg

# matplotlib and tsne
import matplotlib.pyplot as plt
# import visualizeTriplet

# logger
import logging
import time

# ================================================================== #

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imageRootDir', '-i', help='imageRootDir')
    parser.add_argument('--frameLengthCSV', '-c', help='Path_to_frame_length.csv')

    return parser.parse_args()


def plot_graph(train_loss, test_loss, outputPath):
    fig = plt.figure(figsize=(12, 6))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(["train_loss", "test_loss"], loc="upper right")

    plt.plot(range(len(train_loss)), train_loss, "r", linewidth=1.5, linestyle="-")
    plt.plot(range(len(test_loss)), test_loss, "b", linewidth=1.5, linestyle="-")
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
    train_recipeProcessDataset = RecipeProcessDataset(imageRootDir=args.imageRootDir, frameLengthCSV=args.frameLengthCSV, anchorSize=img_size, trainOrTest="train", randomSeed=1)
    test_recipeProcessDataset = RecipeProcessDataset(imageRootDir=args.imageRootDir, frameLengthCSV=args.frameLengthCSV, anchorSize=img_size, trainOrTest="test", randomSeed=1)
    train_loader = torch.utils.data.DataLoader(dataset=train_recipeProcessDataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_recipeProcessDataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # model
    model = make_model('inception_v3', num_classes=3, pretrained=True, input_size=img_size)
    device = torch.device('cuda')
    model = model.to(device)

    # Loss fuction
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    train_loss = []
    test_loss = []
    for epoch_idx in range(1, epochNum+1, 1):
        model.train()
        epochStartTime = time.time()
        interval_loss_sum = 0
        epoch_loss_sum = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            imgs_vec = model(imgs.to(device))
            batch_loss = criterion(imgs_vec, labels.to(device))

            batch_loss.backward()
            optimizer.step()
            scheduler.step()

            interval_loss_sum = interval_loss_sum + batch_loss.item()
            epoch_loss_sum = epoch_loss_sum + batch_loss.item()

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

        # predict test samples
        del imgs_vec
        torch.cuda.empty_cache()

        model.eval()
        epoch_test_loss = 0
        for batch_idx, (imgs, labels) in enumerate(test_loader, start=1):
            with torch.no_grad():
                outputs = model(imgs.to(device))
                batch_loss = criterion(outputs, labels.to(device))

            epoch_test_loss += batch_loss.item()
            del imgs
            torch.cuda.empty_cache()

        epoch_test_loss = epoch_test_loss / batch_idx

        logger.log(30, "#################### Epoch {} summary ######################".format(epoch_idx))
        logger.log(30, "epoch_loss : {}".format(epoch_loss))
        logger.log(30, "test_loss : {}".format(epoch_test_loss))
        train_loss.append(epoch_loss)
        test_loss.append(epoch_test_loss)

        plot_graph(train_loss, test_loss, cfg.GRAPH_PDF)

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
