from pathlib import Path

EPOCH = 500
OPTIMIZER = 'Adam'
LOG_INTERVAL = 200
BATCH_SIZE = 1
NUM_EX = "16"
GRAPH_PDF = "./ex{}/trian_loss.pdf".format(NUM_EX)
TSNE_DEBUG = False
LOSS_MARGIN = 1.0

PARAM_EPOCH = "96"
TRAINED_PARAM = "./ex{}/params_ex{}_epoch{}.ckpt".format(NUM_EX, NUM_EX, PARAM_EPOCH)
TEST_DIR_ROOT = "/home/dataset/recipeVideoDataset/"
TEST_BATCH_SIZE = 24
