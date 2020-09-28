import torch
import torch.nn as nn

rdc_text_dim = 1000
z_dim = 100
h_dim = 4086


class _param:
    def __init__(self):
        self.rdc_text_dim = rdc_text_dim
        self.z_dim = z_dim
        self.h_dim = h_dim


# with double linear layers
class _netGPlus(nn.Module):
    def __init__(self, text_dim=11083, X_dim=3584):
        super(_netGPlus, self).__init__()
        # self.rdc_text = nn.Linear(text_dim, rdc_text_dim)
        self.pre_text = nn.Conv1d(1, text_dim, 3, padding=1)
        self.rdc_text = nn.Conv1d(text_dim, rdc_text_dim, 3, padding=1)
        self.main = nn.Sequential(nn.Conv1d(z_dim + rdc_text_dim, h_dim, 3, padding=1),
                                  nn.LeakyReLU(),
                                  nn.Conv1d(h_dim, h_dim, 3, padding=1),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, X_dim),
                                  nn.Tanh())

    def forward(self, z, c):
        print(z.size(), c.size()) # z: (1000, 100), c: (1000, 7551)
        # z.view(z.size()[0], 1, -1)
        # c.view(c.size()[0], 1, -1)
        # think about how to change this setting
        z.unsqueeze(1)
        c.unsqueeze(1)
        print('========', z.size()[0], z.size(), c.size())
        pre_text = self.pre_text(c)
        rdc_text = self.rdc_text(pre_text)
        exit(0)
        input = torch.cat([z, rdc_text], 1)
        output = self.main(input)
        return output
# reduce to dim of text first
# class _netGPlus(nn.Module):
#     def __init__(self, text_dim=11083, X_dim=3584):
#         super(_netGPlus, self).__init__()
#         self.rdc_text = nn.Linear(text_dim, rdc_text_dim)
#         self.main = nn.Sequential(nn.Linear(z_dim + rdc_text_dim, h_dim),
#                                   nn.LeakyReLU(),
#                                   nn.Linear(h_dim, X_dim),
#                                   nn.Tanh())
#
#     def forward(self, z, c):
#         rdc_text = self.rdc_text(c)
#         input = torch.cat([z, rdc_text], 1)
#         output = self.main(input)
#         return output


class _netDPlus(nn.Module):
    def __init__(self, y_dim=150, X_dim=3584):
        super(_netDPlus, self).__init__()
        # Discriminator net layer one
        self.D_shared = nn.Sequential(nn.Linear(X_dim, h_dim),
                                      nn.ReLU(),
                                      nn.Linear(h_dim, h_dim),
                                      nn.ReLU())
        # Discriminator net branch one: For Gan_loss
        # self.D_gan = nn.Linear(h_dim, 1)
        self.D_gan = nn.Sequential(nn.Linear(h_dim, h_dim),
                                   nn.ReLU(),
                                   nn.Linear(h_dim, 1))
        # Discriminator net branch two: For aux cls loss
        # self.D_aux = nn.Linear(h_dim, y_dim)
        self.D_aux = nn.Sequential(nn.Linear(h_dim, h_dim),
                                   nn.ReLU(),
                                   nn.Linear(h_dim, y_dim))

    def forward(self, input):
        h = self.D_shared(input)
        return self.D_gan(h), self.D_aux(h)


