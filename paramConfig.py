from pathlib import Path

# TRAIN_PARAMETERS
NUM_TRAINEX = "25"
EPOCH = 500
BATCH_SIZE = 1
LOG_INTERVAL = 50 * BATCH_SIZE
GRAPH_PDF = "./ex{}/train_loss.pdf".format(NUM_TRAINEX)
TSNE_DEBUG = False
LOSS_MARGIN = 1.0
IMG_SIZE = (384, 512)
NET_DIMENTIONS = 64
LEARNING_RATE = 0.005
LOSS_FUNCTION = "quintupletLoss"
ENABLE_SWAP = True

# TEST_PARAMETERS
NUM_TESTEX = "24"
PARAM_EPOCH = "96"
TRAINED_PARAM = "./ex{}/params_ex{}_epoch{}.ckpt".format(NUM_TESTEX, NUM_TESTEX, PARAM_EPOCH)
TEST_DIR_ROOT = "/home/dataset/recipeVideoDataset/video_and_frames/"
TEST_BATCH_SIZE = 36
LOGFILE = './log/test_log_ex{}.log'.format(NUM_TESTEX)
TEST_NET_DIMENTIONS = 64
TEST_IMG_SIZE=(384, 512)
