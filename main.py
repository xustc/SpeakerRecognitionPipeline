# speaker recognition pipeline

from config import Config


class Pipeline(object):

    def extract(self):
        from data.extract_features import FeatureExtractor
        extractor = FeatureExtractor()
        extractor()

    def train(self, triplet=True):
        # generate dataset for PyTorch
        from data.dataset import TruncatedInputFromMFB, ToTensor, SpeakerDataset
        from torchvision import transforms
        from torch.utils.data import DataLoader
        import torch
        transform = transforms.Compose([TruncatedInputFromMFB(), ToTensor()])
        if Config.MODEL_TYPE == 'cnn3d':
            from data.dataset3d import SpeakerDataset3D
            initial_dataset = SpeakerDataset(transform=transform)
            train_dataset = SpeakerDataset3D(initial_dataset)
        else:
            train_dataset = SpeakerDataset(transform=transform)

        # instantiate a model
        if Config.MODEL_TYPE == 'rescnn':
            from models.rescnn import ResNet
            model_ = ResNet(layers=Config.RESCNN_LAYERS, num_classes=Config.NUM_CLASSES)
        elif Config.MODEL_TYPE == 'gru':
            from models.gru import GRU
            model_ = GRU(layers=Config.GRU_LAYERS, num_classes=Config.NUM_CLASSES)
        elif Config.MODEL_TYPE == 'cnn3d':
            from models.cnn3d import CNN3D
            model_ = CNN3D(num_classes=Config.NUM_CLASSES)

        from utils.train import Trainer
        model_ = model_.cuda()
        epoch = Config.SOFTMAX_TRAINING_EPOCH
        for i in range(epoch):
            optimizer = torch.optim.Adam(model_.parameters())
            train_loader = DataLoader(train_dataset, batch_size=Config.PRETRAIN_BATCH_SIZE, shuffle=True)
            Trainer.train(train_loader, model_, optimizer, i)

        if triplet:
            from copy import deepcopy
            model_tri = deepcopy(model_)
            model_tri = model_tri.cuda()
            epoch_ = Config.TRIPLET_TRAINING_EPOCH
            for i in range(epoch_):
                optimizer_ = torch.optim.SGD(model_tri.parameters(),
                                             lr=Config.TRIPLET_LR - i * Config.TRIPLET_LR_DECAY,
                                             momentum=Config.TRIPLET_MOMENTUM)
                train_loader = DataLoader(train_dataset, batch_size=Config.FINETUNE_BATCH_SIZE, shuffle=True)
                Trainer.train_tri(train_loader, model_tri, optimizer_, i,
                                  semi_hard=True, triplet_margin=Config.TRIPLET_MARGIN)

    def test(self):
        from utils.test import Test
        test = Test()
        test()

    def export(self):
        from utils.onnx import export_onnx
        export_onnx()


if __name__ == '__main__':
    import fire
    fire.Fire(Pipeline)