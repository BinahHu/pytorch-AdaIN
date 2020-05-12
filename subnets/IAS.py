import torch.nn as nn
import torch.nn.functional as F
import torch

class IAS(nn.Module):
    def __init__(self):
        super(IAS, self).__init__()
        # Linear regression
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 15)
        self.mse_loss = nn.MSELoss()

    def forward(self, x, color5):
        y_pred = self.predict(x)
        y_expt = self.rcs.find_nearest_cord(color5)
        y_expt = torch.from_numpy(y_expt).float().cuda()
        loss = self.mse_loss(y_pred, y_expt)

        return loss

    def predict_cord(self, x):
        x = self.pool(x)
        N = x.shape[0]
        x = x.view(N, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        #y_pred = self.linear3(x)
        y_pred = self.linear3(x)

        return y_pred

    def predict_color(self, x):
        x = self.pool(x)
        N = x.shape[0]
        x = x.view(N, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        #y_pred = self.linear3(x)
        y_pred = F.sigmoid(self.linear3(x))

        return y_pred