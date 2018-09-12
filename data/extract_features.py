# 提取特征

import os
import re
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from python_speech_features import mfcc, fbank
from python_speech_features import delta

from config import Config

TRAIN_DATAPATH = Config.TRAIN_DATA_PATH
TEST_DATAPATH = Config.TEST_DATA_PATH
FEATURE_TYPE = Config.FEATURE_TYPE


class FeatureExtractor(object):

    def __call__(self):
        train_speakers = []
        test_speakers = []

        # 添加 speakers 列表
        for wav_file in os.listdir(TRAIN_DATAPATH):
            if re.match('.*wav$', wav_file):
                speaker = wav_file.split('_')[0]
                if speaker not in train_speakers:
                    train_speakers.append(speaker)

        for wav_file in os.listdir(TEST_DATAPATH):
            if re.match('.*wav$', wav_file):
                speaker = wav_file.split('_')[0]
                if speaker not in test_speakers:
                    test_speakers.append(speaker)

        # 分组提取特征
        for speaker in train_speakers:
            if not os.path.exists(os.path.join(TRAIN_DATAPATH, FEATURE_TYPE, speaker)):
                os.makedirs(os.path.join(TRAIN_DATAPATH, FEATURE_TYPE, speaker))

        for speaker in test_speakers:
            if not os.path.exists(os.path.join(TEST_DATAPATH, FEATURE_TYPE, speaker)):
                os.makedirs(os.path.join(TEST_DATAPATH, FEATURE_TYPE, speaker))

        for wav_file in os.listdir(TRAIN_DATAPATH):
            if re.match('.*wav$', wav_file):
                speaker = wav_file.split('_')[0]
                self.extract_feature(os.path.join(TRAIN_DATAPATH, wav_file),
                                     os.path.join(TRAIN_DATAPATH, FEATURE_TYPE, speaker, wav_file))

        for wav_file in os.listdir(TEST_DATAPATH):
            if re.match('.*wav$', wav_file):
                speaker = wav_file.split('_')[0]
                self.extract_feature(os.path.join(TEST_DATAPATH, wav_file),
                                     os.path.join(TEST_DATAPATH, FEATURE_TYPE, speaker, wav_file))

    @staticmethod
    def extract_feature(wav_path, feature_path):
        if not os.path.exists(feature_path):
            rate, sig = wav.read(wav_path)
            try:
                if FEATURE_TYPE == 'mfcc':
                    mfcc_ = preprocessing.scale(mfcc(sig, samplerate=rate, numcep=40, nfilt=80), axis=1)
                    delta_1 = preprocessing.scale(delta(mfcc_, N=1), axis=1)
                    delta_2 = preprocessing.scale(delta(delta_1, N=1), axis=1)
                    np.save(feature_path, np.hstack([mfcc_, delta_1, delta_2]))  # 将3个 n*40 维特征堆叠成为 n*120 维特征
                elif FEATURE_TYPE == 'fbank':
                    fbank_ = preprocessing.scale(fbank(sig, samplerate=rate, nfilt=80), axis=1)
                    delta_1 = preprocessing.scale(delta(fbank_, N=1), axis=1)
                    delta_2 = preprocessing.scale(delta(delta_1, N=1), axis=1)
                    np.save(feature_path, np.hstack([fbank_, delta_1, delta_2]))
            except ValueError:
                print('only mfcc/fbank is allowed.')
        return
