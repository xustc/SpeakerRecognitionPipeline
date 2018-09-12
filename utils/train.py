## softmax pre-training

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import os

from config import Config
from models.tripletloss import TripletLoss


class Trainer(object):

    @staticmethod
    def train(train_loader, model, optimizer, epoch):
        model.train()  # switch to training mode
        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(enumerate(train_loader))
        for batch_idx, (data, label) in pbar:
            data, label = Variable(data).cuda(), Variable(label).cuda()

            out = model(data)[0].cuda()
            loss = criterion(out, label).cuda()

            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx + 1, len(train_loader),
                           100. * (batch_idx + 1) / len(train_loader),
                    loss.data[0]))

        # save model
        model_name = Config.MODEL_TYPE + '.pth'
        torch.save(model.state_dict(), os.path.join('./checkpoints', model_name))

    @staticmethod
    def train_tri(train_loader, model, optimizer, epoch, semi_hard=True, triplet_margin=0.1):
        def cal_similarity(ts1, ts2):
            from numpy import linalg as LA
            from scipy.spatial.distance import pdist
            ts1_ = (ts1 / LA.norm(ts1))  # .cpu().detach().numpy()
            ts2_ = (ts2 / LA.norm(ts2))  # .cpu().detach().numpy()

            x = np.vstack([ts1_, ts2_])
            d = pdist(x, 'cosine')
            return 1 - d
        model.train()  # switch to train mode

        pbar = tqdm(enumerate(train_loader))
        for batch_idx, (data, label) in pbar:
            data, label = Variable(data).cuda(), Variable(label).cuda()
            embeddings = model(data)[1]

            batch_size = embeddings.shape[0]
            index_list = [[i, j, k] for i in range(batch_size) \
                          for j in range(batch_size) \
                          for k in range(batch_size) if i != j]

            valid_triplets = []
            label_ = label.cpu().numpy()  # for acceleration
            for index in index_list:
                if label_[index[0]] == label_[index[1]] and label_[index[0]] != label_[index[2]]:
                    valid_triplets.append(index)
            print("valid length =", len(valid_triplets))

            # triplet mining
            hard_triplets = []
            normal_triplets = []
            embeddings_ = embeddings.cpu().detach().numpy()  # for acceleration
            for index in valid_triplets:
                sim_ap = cal_similarity(embeddings_[index[0]], embeddings_[index[1]])
                sim_an = cal_similarity(embeddings_[index[0]], embeddings_[index[2]])

                margin_ = triplet_margin
                if semi_hard:
                    if sim_an + margin_ > sim_ap:
                        hard_triplets.append(index)
                    elif sim_an < sim_ap:
                        normal_triplets.append(index)
                else:
                    if sim_an > sim_ap:
                        hard_triplets.append(index)

            if len(normal_triplets) > len(hard_triplets):
                hard_triplets.extend(normal_triplets[:len(hard_triplets)])
            else:
                hard_triplets.extend(normal_triplets)
            print("hard length =", len(hard_triplets))

            anchors_list = []
            positives_list = []
            negatives_list = []
            for triplet in hard_triplets:
                anchors_list.append(embeddings[triplet[0]])
                positives_list.append(embeddings[triplet[1]])
                negatives_list.append(embeddings[triplet[2]])

            triplet_length = len(anchors_list)
            if triplet_length > 0:
                # convert list of 1-D tensor to 2-D tensor
                anchors = torch.cat(anchors_list).reshape((len(anchors_list), 512)).cuda()
                positives = torch.cat(positives_list).reshape((len(positives_list), 512)).cuda()
                negatives = torch.cat(negatives_list).reshape((len(negatives_list), 512)).cuda()

                loss = TripletLoss(margin=triplet_margin).forward(anchors, positives, negatives).cuda()

                # compute gradient and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx + 1, len(train_loader),
                               100. * (batch_idx + 1) / len(train_loader),
                        loss.data[0]))

        # save model
        model_name = Config.MODEL_TYPE + '_tri.pth'
        torch.save(model.state_dict(), os.path.join('./checkpoints', model_name))
