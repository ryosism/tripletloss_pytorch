from pathlib import Path

# TRAIN_PARAMETERS
NUM_TRAINEX = "35"
EPOCH = 500
BATCH_SIZE = 1
LOG_INTERVAL = 50 * BATCH_SIZE
GRAPH_PDF = "./ex{}/train_loss.pdf".format(NUM_TRAINEX)
TSNE_DEBUG = False
LOSS_MARGIN = 1000
IMG_SIZE = (384, 512)
NET_DIMENTIONS = 64
LEARNING_RATE = 0.05
LOSS_FUNCTION = "clusteredTripletLoss"
ENABLE_SWAP = True
NUM_OF_SAMPLES_FOR_SWAP = 9

# INFO: GPUのメモリの都合上1080ti(11GB)では，BATCH_SIZE * NUM_OF_SAMPLES_FOR_SWAP =< 18であることが好ましい(他のハイパラで変わるけど)

# TEST_PARAMETERS
NUM_TESTEX = "32"
PARAM_EPOCH = "96"
TRAINED_PARAM = "./ex{}/params_ex{}_epoch{}.ckpt".format(NUM_TESTEX, NUM_TESTEX, PARAM_EPOCH)
TEST_DIR_ROOT = "/home/dataset/recipeVideoDataset/video_and_frames/"
TEST_BATCH_SIZE = 36
LOGFILE = './log/test_log_ex{}.log'.format(NUM_TESTEX)
TEST_NET_DIMENTIONS = 64
TEST_IMG_SIZE=(384, 512)
TEST_PERPLEXITY = 200
