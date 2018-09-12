# construct dataset3d based on dataset.py

import torch
from torch.utils.data import Dataset


class SpeakerDataset3D(Dataset):

    def __init__(self, train_dataset):
        # find out the num of wav whose frame>400 in each class, among 'train_dataset'
        num_of_class = 0
        utr_in_class = 0
        utr_in_classes = []
        for i in range(len(train_dataset)):
            if train_dataset[i][1] == num_of_class:
                utr_in_class += 1
            else:
                utr_in_classes.append(utr_in_class)
                utr_in_class = 1
                num_of_class += 1
        utr_in_classes.append(utr_in_class)

        new_data = []
        new_label = []
        for idx, utr_in_class in enumerate(utr_in_classes):
            num_data_new = int(utr_in_class / 5)
            if idx == 0:
                prev_num = 0
            else:
                prev_num = utr_in_classes[idx - 1]
            for i in range(num_data_new):  # (3, 40, 400) -> (3, 40, 80, 5)
                tensors_new = [train_dataset[5 * i + prev_num + j][0].reshape((3, 40, 80, 5)) for j in range(4)]
                tensor_new = torch.cat(tensors_new, dim=3)  # 4 * (3, 40, 80, 5) -> (3, 40, 80, 20)
                new_data.append(torch.transpose(tensor_new, 1, 3))  # (3, 40, 80, 20) -> (3, 20, 80, 40)
                new_label.append(idx)

        self.features = new_data
        self.classes = new_label

    def __getitem__(self, index):
        return self.features[index], self.classes[index]

    def __len__(self):
        return len(self.features)
