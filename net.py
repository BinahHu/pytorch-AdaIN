import torch.nn as nn
from skimage import color
from sklearn.cluster import KMeans

from function import adaptive_instance_normalization as adain
from function import adaptive_instance_normalization_cat_color as adain_cat
from function import calc_mean_std
import torch
import torch.nn.functional as F
import numpy as np

from models.decoder import  decoder
from models.vgg import vgg
from models.IAS import IAS
from models.discriminator import Discriminator
from models.generator import unet_generator

class Net(nn.Module):
    def __init__(self, encoder, decoder, args):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.ias = IAS()
        self.G = unet_generator(args.input_channel, args.output_channel, args.n_feats, args.color_feat_dim)
        self.D = Discriminator(args.input_channel + args.output_channel, args.color_feat_dim, args.img_size)

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

    def calc_generate_loss(self):
        return None

    def calc_discriminator_loss(self):
        return None

    def forward_decoder(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t0 = adain(content_feat, style_feats[-1])
        t = alpha * t0 + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], content_feat)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s, g_t

    def get_color_feat(self, imgs):
        color_feat = self.ias.predict_color(imgs)
        return color_feat


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