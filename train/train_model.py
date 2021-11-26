
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class LocNet(nn.Module):
    def __init__(self):
        super(LocNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.linear1 = nn.Linear(327680, 512)
        self.linear2 = nn.Linear(512, 258)
        self.linear3 = nn.Linear(258, 126)
        self.linear4 = nn.Linear(126, 8)

        # self.convnet = nn.Sequential(nn.Conv2d(1, 20, 3), nn.PReLU(), nn.BatchNorm2d(20), # in_channels, out_channels, kernel_size,
        #                             nn.MaxPool2d(3, stride=1),
        #                             nn.Conv2d(20, 40, 3), nn.PReLU(), nn.BatchNorm2d(40),
        #                             nn.MaxPool2d(3, stride=1),
        #                             nn.Conv2d(40, 80, 3), nn.PReLU(), nn.BatchNorm2d(80),
        #                             nn.MaxPool2d(3, stride=1),
        #                             nn.Conv2d(80, 160, 3), nn.PReLU(), nn.BatchNorm2d(160),
        #                             nn.MaxPool2d(3, stride=1))

        # self.fc = nn.Sequential(nn.Linear(21760, 512),
        #                     nn.PReLU(), nn.BatchNorm1d(512),
        #                     nn.Linear(512, 256),
        #                     nn.PReLU(), nn.BatchNorm1d(256),
        #                     nn.Linear(256, 128),
        #                     nn.PReLU(), nn.BatchNorm1d(128),
        #                     nn.Linear(128, 8) 
        #                     )

    def forward(self, x): # 64 x 1 x 16 x 80
        # x = x.view(x.size()[0], -1)
        # print("size: " , x.shape)
        x = torch.reshape(x, (x.size()[0], 1, x.size()[1], x.size()[2]))
        x.cuda()
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = output.view(output.size()[0], -1)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        
        return output

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2): # x: 64 (batch) * 16 (channel) * 80 (bin)
        x1.cuda()
        x2.cuda()
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)
