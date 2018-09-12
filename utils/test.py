# test/inference

import re
import os
import numpy as np
from data.dataset import TruncatedInputFromMFB, ToTensor
from torchvision import transforms
from config import Config


class Test(object):

    def __init__(self):
        # import model
        import torch
        if Config.MODEL_TYPE == 'rescnn':
            from models.rescnn import ResNet
            model_ = ResNet(layers=Config.RESCNN_LAYERS, num_classes=Config.NUM_CLASSES)
        elif Config.MODEL_TYPE == 'gru':
            from models.gru import GRU
            model_ = GRU(layers=Config.GRU_LAYERS, num_classes=Config.NUM_CLASSES)
        elif Config.MODEL_TYPE == 'cnn3d':
            from models.cnn3d import CNN3D
            model_ = CNN3D(num_classes=Config.NUM_CLASSES)
        model_.load_state_dict(torch.load(Config.LOAD_PATH))

        # prepare data for test
        test_datapath_1 = Config.TEST_DATAPATH_1
        test_datapath_2 = Config.TEST_DATAPATH_2
        transform = transforms.Compose([TruncatedInputFromMFB(), ToTensor()])
        test_np_1 = []
        test_np_2 = []
        model4test = model_.cuda()
        for np_file in os.listdir(test_datapath_1[:-1]):
            if re.match('.*npy$', np_file):
                test_np_1.append(transform(np.load(test_datapath_1 + np_file)).reshape((1, 3, 64, 64)))

        for np_file in os.listdir(test_datapath_2[:-1]):
            if re.match('.*npy$', np_file):
                test_np_2.append(transform(np.load(test_datapath_2 + np_file)).reshape((1, 3, 64, 64)))

        self.test_embed1 = [model4test(np_file.cuda())[1].cpu().detach().numpy().reshape(512) for np_file in test_np_1]
        self.test_embed2 = [model4test(np_file.cuda())[1].cpu().detach().numpy().reshape(512) for np_file in test_np_2]

    @ staticmethod
    def plot_distances(embed1, embed2, is_identical=False, threshold=0.8):
        # output:[-1,1], higher value means higher similarity
        if isinstance(embed1, list) and isinstance(embed2, list):
            from scipy.spatial.distance import pdist
            distances = []
            num_similars = 0
            for i in range(len(embed1)):
                for j in range(len(embed2)):
                    if (i != j and is_identical) or not is_identical:  # same speaker / different speakers
                        x = np.vstack([embed1[i], embed2[j]])
                        d = pdist(x, 'cosine')
                        distances.append(1 - d)
                        if 1 - d > threshold:
                            num_similars += 1
            print("number of similars:", num_similars)
            print("number in total:", len(distances))
            print("similar triplets percentage:", float(num_similars) / len(distances))
        return

    def __call__(self):
        test_embed1_part = self.test_embed1[:-1]
        test_embed2_part = self.test_embed2[:-1]
        self.plot_distances(test_embed1_part, test_embed1_part, True)
        self.plot_distances(test_embed1_part, test_embed2_part, False)
