import torch.nn as nn
from function import calc_mean_std
import torch

from subnets.IAS import IAS
from subnets.vgg import vgg
from subnets.generator import unet_generator

from util import torch_lab2rgb

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        #vgg.load_state_dict(torch.load(args.vgg))
        #self.encoder = nn.Sequential(*list(vgg.children())[:31])
        #for name in ['encoder']:
        #    for param in getattr(self, name).parameters():
        #        param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.ias = IAS()

        self.L_path = unet_generator(1, 1, args.n_feats)
        self.AB_path = unet_generator(2, 2, args.n_feats)

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        #assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        #assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def get_color_feat(self, imgs):
        color_feat = self.ias.predict_color(imgs)
        return color_feat

    def c_s_loss(self, pred_l, content_l, style_l):
        input_feats = self.L_path.encode_with_intermediate(pred_l)
        target_c = self.L_path.encode(content_l)
        target_s = self.L_path.encode_with_intermediate(style_l)

        loss_c = self.calc_content_loss(input_feats[-1], target_c)
        loss_s = self.calc_style_loss(input_feats[0], target_s[0])
        for i in range(1, len(input_feats) - 1):
            loss_s += self.calc_style_loss(input_feats[i], target_s[i])

        return loss_c, loss_s

    def a_loss(self, pred_ab, aest_ab):
        input_a = self.AB_path.encode_with_intermediate(pred_ab)
        target_a = self.AB_path.encode_with_intermediate(aest_ab)

        loss_a = self.calc_style_loss(input_a[0], target_a[0])
        for i in range(1, len(input_a) - 1):
            loss_a += self.calc_style_loss(input_a[i], target_a[i])

        return loss_a

    def a_loss_true(self, pred_ab, aest_ab):
        input_cord = self.ias.predict_color(pred_ab)
        target_cord = self.ias.predict_color(aest_ab)

        loss_a = self.mse_loss(input_cord, target_cord)
        return loss_a

    def forward(self, content_l, content_ab, style_l, aest_ab, debug=False):
        l_pred = self.L_path(content_l, style_l)
        ab_pred = self.AB_path(content_ab, aest_ab)


        if debug:
            cl_f = self.L_path.encode(content_l)
            sl_f = self.L_path.encode(style_l)
            print(cl_f[0][0])
            print(sl_f[0][0])
            print("Show content and style L feat")
            c = input()

            cab_f = self.AB_path.encode(content_ab)
            aab_f = self.AB_path.encode(aest_ab)
            print(cab_f[0][0])
            print(aab_f[0][0])
            print("Show content and aest AB feat")
            c = input()


            print(l_pred[0])
            print(ab_pred[0])
            print("Show l_pred and ab_pred")
            c = input()

        return l_pred, ab_pred