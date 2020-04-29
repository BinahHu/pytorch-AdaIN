from skimage.color import rgb2lab, lab2rgb
import numpy as np

def adjust_learning_rate(opts, iteration_count, args):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for opt in opts:
        for param_group in opt.param_groups:
            param_group['lr'] = lr

def zero_grad(opts):
    for opt in opts:
        opt.zero_grad()

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