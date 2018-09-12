# triplet loss function

import torch
import torch.nn as nn
from torch.autograd import Function


class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        s_p = cos(anchor, positive)
        s_n = cos(anchor, negative)

        # we utilize cosine similarity instead of L2 norm as distance
        sim_hinge = torch.clamp(self.margin + s_n - s_p, min=0.0)
        loss = torch.mean(sim_hinge)
        return loss
