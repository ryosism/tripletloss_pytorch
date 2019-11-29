from pathlib import Path

NUM_TRAINEX = "21"
EPOCH = 500
OPTIMIZER = 'Adam'
LOG_INTERVAL = 200
BATCH_SIZE = 4
GRAPH_PDF = "./ex{}/train_loss.pdf".format(NUM_TRAINEX)
TSNE_DEBUG = False
LOSS_MARGIN = 1.0
IMG_SIZE = (384, 512)
NET_DIMENTIONS = 64

NUM_TESTEX = "19"
PARAM_EPOCH = "96"
TRAINED_PARAM = "./ex{}/params_ex{}_epoch{}.ckpt".format(NUM_TESTEX, NUM_TESTEX, PARAM_EPOCH)
TEST_DIR_ROOT = "/home/dataset/recipeVideoDataset/video_and_frames/"
TEST_BATCH_SIZE = 24
LOGFILE = './log/test_log_ex{}.log'.format(NUM_TESTEX)
TEST_NET_DIMENTIONS = 128
TEST_IMG_SIZE=(768, 1024)
