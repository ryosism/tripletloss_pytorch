from pathlib import Path

EPOCH = 200
OPTIMIZER = 'Adam'
LOG_INTERVAL = 200
BATCH_SIZE = 1
NUM_EX = "03"
GRAPH_PDF = "./ex{}/trian_loss.pdf".format(NUM_EX)

PARAM_EPOCH = "96"
TRAINED_PARAM = "./ex{}/params_ex{}_epoch{}.ckpt".format(NUM_EX, NUM_EX, PARAM_EPOCH)
TEST_DIR_ROOT = "/home/dataset/recipeVideoDataset/"
TEST_BATCH_SIZE = 24
