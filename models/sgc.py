# semantic guided matrix

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import  numpy as np
import models.util as util

class SGC:
    # train_Y is interger
    def __init__(self, _train_X, _train_Y, _text_feat, _nclass, _input_dim, _cuda=True, _lr=0.001, _beta1=0.5, _nepoch=1000,
                 _batch_size=100, pretrain_classifer=''):
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.text_feat_raw = _text_feat
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _input_dim
        self.cuda = _cuda
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.text_feat_raw.shape[1])
        self.model.apply(util.weights_init)
        self.criterion = nn.LogSoftmax()

        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)
        self.text_feat = torch.FloatTensor(_batch_size, self.text_feat_raw.shape[1])

        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))


        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
            self.text_feat = self.text_feat.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if pretrain_classifer == '':
            self.fit()
        else:
            self.model.load_state_dict(torch.load(pretrain_classifier))

    def fit(self):
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label, batch_text = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                # print(f'this is {batch_text.shape}') # (0)
                self.text_feat.copy_(batch_text) # (100, 7551)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                textv = Variable(self.text_feat)
                output = self.model(inputv) # [batch_size, text_dim]
                # Sc = torch.dot(output, self.text_feat)
                # print(output.shape, batch_text.shape) # (100, 7551), (100, 7551)
                Sc = torch.bmm(output.view(output.shape[0], 1, output.shape[1]), batch_text.view(batch_text.shape[0], batch_text.shape[1], 1))
                loss = self.criterion(Sc)
                loss = torch.mean(-loss)
                # print(f'Current loss is: {loss}')
                loss.backward()
                self.optimizer.step()

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            self.text_feat_raw = self.text_feat_raw[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
                Text_rest_part = self.text_feat_raw[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            self.text_feat_raw = self.text_feat_raw[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            Text_new_part = self.text_feat_raw[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0), torch.cat((Text_rest_part, Text_new_part), 0)
            else:
                return X_new_part, Y_new_part, Text_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end], self.text_feat_raw[start:end]

    # test_label is integer
    def val(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda(), volatile=True))
            else:
                output = self.model(Variable(test_X[start:end], volatile=True))
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label,
                                         target_classes.size(0))
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
        return acc_per_class.mean()


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, text_dim):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        # self.fc = nn.Linear(input_dim, nclass)
        self.fc = nn.Linear(input_dim, text_dim)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o
