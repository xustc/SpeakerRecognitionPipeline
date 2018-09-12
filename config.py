# configuration file


class Config(object):
    TRAIN_DATA_PATH = './data_thchs30/train'
    TEST_DATA_PATH = './data_thchs30/test'

    SOFTMAX_TRAINING_EPOCH = 10
    TRIPLET_TRAINING_EPOCH = 15

    MODEL_TYPE = 'rescnn'
    FEATURE_TYPE = 'mfcc'

    NUM_PREVIOUS_FRAME = 10

    if MODEL_TYPE == 'cnn3d':
        NUM_NEXT_FRAME = 390
        FILTER_BANK = 40  # feature dimension
    else:
        NUM_NEXT_FRAME = 56
        FILTER_BANK = 64

    NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME  # time dimension

    PRETRAIN_BATCH_SIZE = 32
    FINETUNE_BATCH_SIZE = 64

    RESCNN_LAYERS = [3, 3, 3, 3]
    GRU_LAYERS = 3
    NUM_CLASSES = 50

    TEST_DATAPATH_1 = "./data_thchs30/test/mfcc/D11/"
    TEST_DATAPATH_2 = "./data_thchs30/test/mfcc/D12/"

    LOAD_PATH = "./checkpoints/" + MODEL_TYPE + '.pth'
    # LOAD_PATH = "./checkpoints/" + MODEL_TYPE + '_tri.pth'

    TRIPLET_LR = 0.05
    TRIPLET_LR_DECAY = 0.003
    TRIPLET_MOMENTUM = 0.9
    TRIPLET_MARGIN = 0.1
