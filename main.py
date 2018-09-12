# speaker recognition pipeline



# 1. 从原始数据中提取特征
# from data.extract_features import FeatureExtractor
#
# extractor = FeatureExtractor()
# extractor()


# 2. 生成dataset
# from data.dataset import TruncatedInputFromMFB, ToTensor, SpeakerDataset
# from torchvision import transforms
# from torch.utils.data import DataLoader
#
# transform = transforms.Compose([TruncatedInputFromMFB(), ToTensor()])
# if Config.MODEL_TYPE == 'cnn3d':
#     from data.dataset3d import SpeakerDataset3D
#
#     initial_dataset = SpeakerDataset(transform=transform)
#     train_dataset = SpeakerDataset3D(initial_dataset)
# else:
#     train_dataset = SpeakerDataset(transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=Config.PRETRAIN_BATCH_SIZE, shuffle=True)

# 3. 实例化模型
# if Config.MODEL_TYPE == 'rescnn':
#     from models.rescnn import ResNet
#
#     model_ = ResNet(layers=Config.RESCNN_LAYERS, num_classes=Config.NUM_CLASSES)
# elif Config.MODEL_TYPE == 'gru':
#     from models.gru import GRU
#
#     model_ = GRU(layers=Config.GRU_LAYERS, num_classes=Config.NUM_CLASSES)
# elif Config.MODEL_TYPE == 'cnn3d':
#     from models.cnn3d import CNN3D
#
#     model_ = CNN3D(num_classes=Config.NUM_CLASSES)

# 4. 训练模型(pre-train)

# 5. 训练模型(triplet)

# 6. 测试

# 7. 部署上线


if __name__ == '__main__':
    from config import Config

