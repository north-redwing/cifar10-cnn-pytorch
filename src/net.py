import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, (3,3))
        self.conv3 = nn.Conv2d(32, 64, (3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, (3,3))
        self.fc1 = nn.Linear(6 * 6 * 64, 512)
        self.fc2 = nn.Linear(512, 10)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.50)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = x.view(-1, 6 * 6 * 64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x