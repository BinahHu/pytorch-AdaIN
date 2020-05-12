from skimage.color import rgb2lab, lab2rgb
import numpy as np
import torch

def adjust_learning_rate(opts, iteration_count, args):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for opt in opts:
        for param_group in opt.param_groups:
            param_group['lr'] = lr

def my_rgb2lab(rgb_image):
    rgb_image = np.transpose(rgb_image, (1,2,0))
    lab_image = rgb2lab(rgb_image)
    l_image = np.transpose(lab_image[:,:,:1], (2,0,1))
    ab_image = np.transpose(lab_image[:,:,1:], (2,0,1))
    return l_image, ab_image

def my_lab2rgb(lab_image):
    lab_image = np.transpose(lab_image, (1,2,0))
    rgb_image = lab2rgb(lab_image)
    rgb_image = np.transpose(rgb_image, (2,0,1))
    return rgb_image

def res_lab2rgb(l, ab):
    l = l.cpu().numpy()
    ab = ab.cpu().numpy()

    l = l * (100.0 + 0.0) - 0.0
    a = ab[0:1] * (98.0  + 86.0) - 86.0
    b = ab[1:2] * (94.0 + 107.0) - 107.0

    lab = np.concatenate((l, a, b), axis=0)
    lab = np.transpose(lab, (1, 2, 0))
    rgb = lab2rgb(lab)
    rgb = (np.array(rgb) * 255).astype(np.uint8)
    return rgb, l[0]



## The following lab2rgb function is credited by RAY075HL
## Original code url: https://ray075hl.github.io/ray075hl.github.io/Pytorch-LAB2RGB/

def torch_lab2rgb(lab_image):
    '''
    :param lab_image: [0., 1.]  format: Bx3XHxW
    :return: rgb_image [0., 1.]
    '''

    # rgb_image = torch.zeros(lab_image.size()).to(config.device)
    # rgb_image = torch.zeros(lab_image.size()).cuda()

    l_s = lab_image[:, 0, :, :] * (100.0 + 0.0) - 0.0
    a_s = lab_image[:, 1, :, :] * (98.0 + 86.0) - 86.0
    b_s = lab_image[:, 2, :, :] * (94.0 + 107.0) - 107.0

    var_Y = (l_s + 16.0) / 116.
    var_X = a_s / 500. + var_Y
    var_Z = var_Y - b_s / 200.

    mask_Y = var_Y.pow(3.0) > 0.008856
    mask_X = var_X.pow(3.0) > 0.008856
    mask_Z = var_Z.pow(3.0) > 0.008856

    Y_1 = var_Y.pow(3.0) * mask_Y.float()
    Y_2 = (var_Y - 16. / 116.) / 7.787 * (~mask_Y).float()
    var_Y = Y_1 + Y_2

    X_1 = var_X.pow(3.0) * mask_X.float()
    X_2 = (var_X - 16. / 116.) / 7.787 * (~mask_X).float()
    var_X = X_1 + X_2

    Z_1 = var_Z.pow(3.0) * mask_Z.float()
    Z_2 = (var_Z - 16. / 116.) / 7.787 * (~mask_Z).float()
    var_Z = Z_1 + Z_2

    X = 0.95047 * var_X
    Y = 1.00000 * var_Y
    Z = 1.08883 * var_Z

    var_R = X * 3.2406 + Y * -1.5372 + Z * -0.4986
    var_G = X * -0.9689 + Y * 1.8758 + Z * 0.0415
    var_B = X * 0.0557 + Y * -0.2040 + Z * 1.0570

    mask_R = var_R > 0.0031308
    R_1 = (1.055 * var_R.pow(1 / 2.4) - 0.055) * mask_R.float()
    R_2 = (12.92 * var_R) * (~mask_R).float()
    var_R = R_1 + R_2

    mask_G = var_G > 0.0031308
    G_1 = (1.055 * var_G.pow(1 / 2.4) - 0.055) * mask_G.float()
    G_2 = (12.92 * var_G) * (~mask_G).float()
    var_G = G_1 + G_2

    mask_B = var_B > 0.0031308
    B_1 = (1.055 * var_B.pow(1 / 2.4) - 0.055) * mask_B.float()
    B_2 = (12.92 * var_B) * (~mask_B).float()
    var_B = B_1 + B_2


    return torch.stack((var_R.unsqueeze(1),
                        var_G.unsqueeze(1),
                        var_B.unsqueeze(1)), dim=1).clamp(0., 1.)