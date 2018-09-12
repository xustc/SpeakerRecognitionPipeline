# GRU architecture

import torch
import torch.nn as nn


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'


class GRU(nn.Module):
    """
    input:  n * channels(3) * freq(64) * frame(64)
    output: n * num_classes
    """

    def __init__(self, layers, num_classes):

        super(GRU, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = ReLU(inplace=True)

        self.gru = nn.GRU(input_size=2048, hidden_size=1024, num_layers=layers, bias=False, batch_first=True)
        self.avg_pool = nn.AvgPool2d([32, 1])  # average on 'frame'

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax()

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                nn.init.orthogonal_(m.weight_hh_l0)
                nn.init.orthogonal_(m.weight_ih_l0)
                nn.init.orthogonal_(m.weight_hh_l1)
                nn.init.orthogonal_(m.weight_ih_l1)
                nn.init.orthogonal_(m.weight_hh_l2)
                nn.init.orthogonal_(m.weight_ih_l2)

    def forward(self, inp):
        out = self.conv(inp)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view((out.size(0), -1, out.size(3)))  # (n, 2048, 32)
        out = torch.transpose(out, 1, 2)

        out, _ = self.gru(out)
        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        embed = out  # (n, 512)

        out = self.fc2(out)  # (n, classes)
        return out, embed
