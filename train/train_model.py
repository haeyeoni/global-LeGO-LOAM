
import torch.nn as nn


class LocNet(nn.Module):
    def __init__(self):
        super(LocNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(2, 50, 3), nn.PReLU(),
                                    nn.MaxPool2d(3, stride=1),
                                    nn.Conv2d(50, 100, 3), nn.PReLU(),
                                    nn.MaxPool2d(3, stride=1))

        self.fc = nn.Sequential(nn.Linear(57600, 256),
                            nn.PReLU(),
                            nn.Linear(256, 128),
                            nn.PReLU(),
                            nn.Linear(128, 3)
                            )

    def forward(self, x): # 1 x 2 x 64 x 80
        # x = x.view(x.size()[0], -1)
        # print("size: " , x.shape)
        x.cuda()
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        x1.cuda()
        x2.cuda()
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)
