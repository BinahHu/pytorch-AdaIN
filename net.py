import torch.nn as nn
from function import calc_mean_std
from function import adaptive_instance_normalization as adain
import torch

from subnets.vgg import vgg
from subnets.decoder import decoder_L, decoder_AB


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        vgg.load_state_dict(torch.load(args.vgg))
        encoder = nn.Sequential(*list(vgg.children())[:31])
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.L_path = decoder_L
        self.AB_path = decoder_AB

        self.mse_loss = nn.MSELoss()

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
        #assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        #assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def c_s_loss(self, pred_l, content_l, style_l):
        pred_l = pred_l.repeat(1, 3, 1, 1)
        input_feats = self.encode_with_intermediate(pred_l)
        target_c = self.encode(content_l)
        target_s = self.encode_with_intermediate(style_l)

        loss_c = self.calc_content_loss(input_feats[-1], target_c)
        loss_s = self.calc_style_loss(input_feats[0], target_s[0])
        for i in range(1, len(input_feats) - 1):
            loss_s += self.calc_style_loss(input_feats[i], target_s[i])

        return loss_c, loss_s

    def a_loss(self, pred_ab, aest_ab):
        zero = torch.zeros(pred_ab.shape[0], 1, pred_ab.shape[2], pred_ab.shape[3]).cuda()
        pred_ab = torch.cat([zero, pred_ab], dim=1)

        input_a = self.encode_with_intermediate(pred_ab)
        target_a = self.encode_with_intermediate(aest_ab)

        loss_a = self.calc_style_loss(input_a[0], target_a[0])
        for i in range(1, len(input_a) - 1):
            loss_a += self.calc_style_loss(input_a[i], target_a[i])

        return loss_a

    def forward(self, content_l, content_ab, style_l, aest_ab, debug=False):
        c_l_feat = self.encode(content_l)
        s_l_feat = self.encode(style_l)

        c_ab_feat = self.encode(content_ab)
        a_ab_feat = self.encode(aest_ab)

        t_l_feat = adain(c_l_feat, s_l_feat)
        t_ab_feat = adain(c_ab_feat, a_ab_feat)

        l_pred = self.L_path(t_l_feat)
        ab_pred = self.AB_path(t_ab_feat)


        if debug:
            cl_f = self.encode(content_l)
            sl_f = self.encode(style_l)
            print(cl_f[0][0])
            print(sl_f[0][0])
            print("Show content and style L feat")
            c = input()

            cab_f = self.encode(content_ab)
            aab_f = self.encode(aest_ab)
            print(cab_f[0][0])
            print(aab_f[0][0])
            print("Show content and aest AB feat")
            c = input()


            print(l_pred[0])
            print(ab_pred[0])
            print("Show l_pred and ab_pred")
            c = input()

        return l_pred, ab_pred