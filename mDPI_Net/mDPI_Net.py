import torch
from fontTools.ttLib.tables.F__e_a_t import Feature
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
class InceptionBlock(nn.Module):

    def __init__(self, in_channels, c1, c2, c3, c4, c5,**kwargs):
        super(InceptionBlock, self).__init__(**kwargs)

        self.p1_1 = nn.Conv3d(in_channels, c1, kernel_size=1)

        self.p2_1 = nn.Conv3d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv3d(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv3d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv3d(c3[0], c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv3d(in_channels, c4, kernel_size=1)

        self.p5_1 = nn.Conv3d(in_channels, c5[0], kernel_size=1)
        self.p5_2 = nn.Conv3d(c5[0], c5[1], kernel_size=7, padding=3)


    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        p5 = F.relu(self.p5_2(F.relu(self.p5_1(x))))

        return torch.cat((p1, p2, p3, p4, p5), dim=1)

class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv3d(1, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            InceptionBlock(96, 32, (48, 64), (8, 16), 16, (4,8)),
            nn.BatchNorm3d(136),
            InceptionBlock(136, 64, (64, 96), (16, 48), 32, (8,24)),
            nn.BatchNorm3d(264),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            InceptionBlock(264, 96, (48, 104), (8, 24), 32, (4,12)),
            nn.BatchNorm3d(268),
            InceptionBlock(268, 80, (56, 112), (12, 32), 32, (6,16)),
            nn.BatchNorm3d(272),
            InceptionBlock(272, 128, (80, 160), (16, 64), 64, (8,32)),
            nn.BatchNorm3d(448),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            InceptionBlock(448, 128, (80, 160), (16, 64), 64, (8,32)),
            nn.BatchNorm3d(448),
            InceptionBlock(448, 128, (96, 192), (24, 64), 64, (12,32)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(480, 128)  # Fully connected layer for classification output
        self.fc2 = nn.Linear(128, 3)  # Fully connected layer for classification output
        self.bn = nn.BatchNorm1d(128)
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x



class FeatureBranch(nn.Module):
    def __init__(self):
        super(FeatureBranch, self).__init__()

        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, padding=1)  
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0) 
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=0)  
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.fc1 = nn.Linear(64 * 15 * 15, 128)
        self.fc2 = nn.Linear(128, 3)


        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)


        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x




class ExInceptionNet(nn.Module):
    def __init__(self):
        super(ExInceptionNet, self).__init__()

        self.inception = InceptionNet()
        self.feature_branch = FeatureBranch()

        self.fc1 = nn.Linear(3, 256)
        self.bn1 = nn.BatchNorm1d(256)  
        self.dropout1 = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(256, 3)

        self.alpha = nn.Parameter()
        self.beta = nn.Parameter()

    def forward(self, x1, x2):

        inception_out = self.inception(x1)
        feature_branch_out = self.feature_branch(x2)
        weighted_out = self.alpha * inception_out + self.beta * feature_branch_out

        return weighted_out



class CustomDataset(Dataset):
    def __init__(self, X, X_fts, y):
        self.X = X  
        self.X_fts = X_fts  
        self.y = y  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x1 = self.X[idx]  
        x2 = self.X_fts[idx]  
        label = self.y[idx]  
        return x1, x2, label  

