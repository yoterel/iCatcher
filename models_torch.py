import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class Encoder_box(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(F.relu(self.bn(x)))
        x = self.fc2(x)

        return x


class Predictor_fc(nn.Module):
    def __init__(self, n, add_box):
        super().__init__()
        in_channel = 512 * n if add_box else 256 * n
        self.fc1 = nn.Linear(in_channel, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x.view(x.size(0), -1))
        x = self.dropout1(F.relu(self.bn1(x)))
        x = self.fc2(x)
        x = self.dropout2(F.relu(self.bn2(x)))
        x = self.fc3(x)
        return x


class GazeCodingModel(nn.Module):
    def __init__(self, device, n, add_box):
        super().__init__()
        self.n = n
        self.device = device
        self.add_box = add_box
        self.encoder_img = resnet18(num_classes=256)
        self.encoder_box = Encoder_box()
        self.predictor = Predictor_fc(n, add_box)

    def forward(self, data):
        imgs = data['imgs'].to(self.device)  # bs x n x 3 x 100 x 100
        boxs = data['boxs'].to(self.device)  # bs x n x 5
        embedding = self.encoder_img(imgs.view(-1, 3, 100, 100)).view(-1, self.n, 256)
        if self.add_box:
            box_embedding = self.encoder_box(boxs.view(-1, 5)).view(-1, self.n, 256)
            embedding = torch.cat([embedding, box_embedding], -1)
        pred = self.predictor(embedding)
        return pred
