
import torch.nn as nn
import numpy as np
import torch


class LocNet(nn.Module):
    def __init__(self):
        super(LocNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 20, 3), nn.PReLU(), # in_channels, out_channels, kernel_size,
                                    nn.MaxPool2d(3, stride=1),
                                    nn.Conv2d(20, 40, 3), nn.PReLU(),
                                    nn.MaxPool2d(3, stride=1),
                                    nn.Conv2d(40, 80, 3), nn.PReLU(),
                                    nn.MaxPool2d(3, stride=1))

        self.fc = nn.Sequential(nn.Linear(21760, 512),
                            nn.PReLU(),
                            nn.Linear(512, 256),
                            nn.PReLU(),
                            nn.Linear(256, 128),
                            nn.PReLU(),
                            nn.Linear(128, 8)
                            )

    def forward(self, x): # 64 x 1 x 16 x 80
        # x = x.view(x.size()[0], -1)
        # print("size: " , x.shape)
        x = torch.reshape(x, (x.size()[0], 1, x.size()[1], x.size()[2]))
        x.cuda()
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
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
