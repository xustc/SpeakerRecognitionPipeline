# CNN3D architecture

import torch.nn as nn


class CNN3D(nn.Module):
    """
    input:  n * channels(3) * uttr(20) * frame(80) * freq(40)
    output: n * num_classes
    """

    def __init__(self, num_classes):
        super(CNN3D, self).__init__()
        self.conv1_1 = nn.Conv3d(3, 16, kernel_size=(3, 1, 5), stride=(1, 1, 1), padding=0, bias=False)
        self.conv1_2 = nn.Conv3d(16, 16, kernel_size=(3, 9, 1), stride=(1, 2, 1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2_1 = nn.Conv3d(16, 32, kernel_size=(3, 1, 4), stride=(1, 1, 1), padding=0, bias=False)
        self.conv2_2 = nn.Conv3d(32, 32, kernel_size=(3, 8, 1), stride=(1, 2, 1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3_1 = nn.Conv3d(32, 64, kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=0, bias=False)
        self.conv3_2 = nn.Conv3d(64, 64, kernel_size=(3, 7, 1), stride=(1, 1, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(64)

        self.conv4_1 = nn.Conv3d(64, 128, kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=0, bias=False)
        self.conv4_2 = nn.Conv3d(128, 128, kernel_size=(3, 7, 1), stride=(1, 1, 1), padding=0, bias=False)
        self.bn4 = nn.BatchNorm3d(128)

        self.relu = nn.PReLU()
        self.avg_pool = nn.AvgPool3d([1, 1, 2])  # average on 'freq'

        self.fc1 = nn.Linear(128 * 4 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inp):
        out = self.conv1_1(inp)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1_2(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.avg_pool(out)

        out = self.conv2_1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2_2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avg_pool(out)

        out = self.conv3_1(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3_2(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4_1(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv4_2(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        embed = out  # (n, 128)

        out = self.fc2(out)  # (n, classes)
        return out, embed
