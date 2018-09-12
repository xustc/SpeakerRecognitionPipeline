# construct dataset

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config

NUM_PREVIOUS_FRAME = Config.NUM_PREVIOUS_FRAME
NUM_NEXT_FRAME = Config.NUM_NEXT_FRAME
NUM_FRAMES = Config.NUM_FRAMES
FILTER_BANK = Config.FILTER_BANK
TRAIN_FEATURE_DIR = os.path.join(Config.TRAIN_DATA_PATH, Config.FEATURE_TYPE)


class TruncatedInputFromMFB(object):

    def __init__(self, input_per_file=1):

        super(TruncatedInputFromMFB, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):

        network_inputs = []
        num_frames = len(frames_features)
        import random

        for i in range(self.input_per_file):
            # select features randomly on time dimension
            j = random.randrange(NUM_PREVIOUS_FRAME, num_frames - NUM_NEXT_FRAME)
            if not j:
                frames_slice = np.zeros(NUM_FRAMES, FILTER_BANK, 'float64')
                frames_slice[0:frames_features.shape[0]] = frames_features.shape
            else:
                frames_slice = frames_features[j - NUM_PREVIOUS_FRAME:j + NUM_NEXT_FRAME]
            network_inputs.append(frames_slice)

        return np.array(network_inputs)


class ToTensor(object):

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            out = torch.FloatTensor(inp.transpose((0, 2, 1)))  # numpy to tensor
            out = torch.cat(torch.chunk(out, 3, dim=1))  # tensor slicing
            return out


class SpeakerDataset(Dataset):

    def __init__(self, num_classes=None, transform=None):
        self.training = (num_classes is None)
        self.transform = transform
        self.features = []
        self.pairID = []
        if self.training:  # construct dataset
            self.classes = []
            self.num_classes = 0
            self.class_id_table = []  # classes corresponds with id_table
            for speaker in os.listdir(TRAIN_FEATURE_DIR):
                self.class_id_table.append(speaker)
                train_feature_subdir = os.path.join(TRAIN_FEATURE_DIR, speaker)
                if os.path.isdir(train_feature_subdir):
                    for feature_file in os.listdir(train_feature_subdir):
                        if feature_file[0] != '.':
                            self.features.append(os.path.join(train_feature_subdir, feature_file))
                            self.classes.append(self.num_classes)
                self.num_classes += 1
            else:
                self.pairID = []  # not done yet

    def __getitem__(self, index):
        if self.training:
            feature = self.transform(np.load(self.features[index]))  # pre-processing
            return feature, self.classes[index]  # return a tuple contains (feature, label)
        else:
            return self.pairID[index], \
                   self.transform(np.load(self.features[index][0])), self.transform(np.load(self.features[index][1]))

    def __len__(self):
        return len(self.features)
