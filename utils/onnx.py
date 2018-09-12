# export ONNX model

import torch.onnx
from torch.autograd import Variable
from config import Config
import os


def export_onnx():
    if Config.MODEL_TYPE == 'rescnn':
        from models.rescnn import ResNet
        model_ = ResNet(layers=Config.RESCNN_LAYERS, num_classes=Config.NUM_CLASSES)
        model_.load_state_dict(torch.load('./checkpoints/...'))  #
        dummy_input = Variable(torch.randn(1, 3, 64, 64))
    elif Config.MODEL_TYPE == 'gru':
        from models.gru import GRU
        model_ = GRU(layers=Config.GRU_LAYERS, num_classes=Config.NUM_CLASSES)

        dummy_input = Variable(torch.randn(1, 3, 64, 64))
    elif Config.MODEL_TYPE == 'cnn3d':
        from models.cnn3d import CNN3D
        model_ = CNN3D(num_classes=Config.NUM_CLASSES)

        dummy_input = Variable(torch.randn(1, 3, 20, 80, 40))

    model_name = Config.MODEL_TYPE + '.onnx'
    torch.onnx.export(model_, dummy_input, os.path.join('./checkpoints', model_name))