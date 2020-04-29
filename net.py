import torch.nn as nn
from skimage import color
from sklearn.cluster import KMeans

from function import adaptive_instance_normalization as adain
from function import adaptive_instance_normalization_cat_color as adain_cat
from function import calc_mean_std
import torch
import torch.nn.functional as F
import numpy as np

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512*2, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class RCS():
    def __init__(self, color_space_file):
        self.CS = self.load_color_space(color_space_file)

    def load_color_space(self, color_space_file):
        data_frame = pd.read_excel(color_space_file, sheet_name='Sheet1')
        data = data_frame.values

        X = data[:, 0:15]
        y = data[:, 15:17]
        K = 5
        #knn = KNeighborsRegressor(K)
        knn.fit(X, y)

        return knn

    def find_nearest_cord(self, x):
        return self.CS.predict(x)

class IAS(nn.Module):
    def __init__(self):
        super(IAS, self).__init__()
        # Linear regression
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 15)
        #self.rcs = RCS(color_space_file)
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


ias = IAS()

class Net(nn.Module):
    def __init__(self, encoder, decoder, ias):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.ias = ias

        # fix the encoder and ias
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'ias']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_aest_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_cord = self.ias.predict_color(input)
        target_cord = self.ias.predict_color(target)

        return self.mse_loss(input_cord, target_cord)

    def forward(self, content, style, aest, alpha=1.0):
    #def forward(self, content, aest, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        aest_feat = self.encode(aest)
        aest_color = self.ias.predict_color(aest_feat)
        t = adain_cat(content_feat, style_feats[-1], aest_feat)
        #t = adain_cat(content_feat, aest_feat)
        #t0 = adain(content_feat, style_feats[-1])
        #t0 = alpha * t0 + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)


        loss_c = self.calc_content_loss(g_t_feats[-1], content_feat)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        loss_a = self.calc_aest_loss(g_t_feats[-1], aest_feat)
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        #return loss_c, loss_a
        return loss_c, loss_s, loss_a