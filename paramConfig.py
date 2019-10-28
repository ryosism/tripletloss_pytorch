from pathlib import Path

NUM_TRAINEX = "18"
EPOCH = 500
OPTIMIZER = 'Adam'
LOG_INTERVAL = 200
BATCH_SIZE = 1
GRAPH_PDF = "./ex{}/train_loss.pdf".format(NUM_TRAINEX)
TSNE_DEBUG = False
LOSS_MARGIN = 1.0

NUM_TESTEX = "11"
PARAM_EPOCH = "96"
TRAINED_PARAM = "./ex{}/params_ex{}_epoch{}.ckpt".format(NUM_TESTEX, NUM_TESTEX, PARAM_EPOCH)
TEST_DIR_ROOT = "/home/dataset/recipeVideoDataset/"
TEST_BATCH_SIZE = 24
LOGFILE = './log/test_log_ex{}.log'.format(NUM_TESTEX)
