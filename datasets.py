from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils import data
import numpy as np
from skimage.color import rgb2lab
import torch

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform(img_size):
    transform_list = [
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

class ColorDataset(data.Dataset):
    def __init__(self, root, img_size, gray_only = False, return_rgb = False):
        super(ColorDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.img_size = img_size
        self.gray_only = gray_only
        self.return_rgb = return_rgb

    def __getitem__(self, index):
        path = self.paths[index]
        rgb_image = Image.open(str(path)).convert('RGB')
        if self.return_rgb:
            transform = train_transform(self.img_size)
            rgb_image_return = train_transform(rgb_image)
        w, h = rgb_image.size
        if w != h:
            min_val = min(w, h)
            rgb_image = rgb_image.crop((w // 2 - min_val // 2, h // 2 - min_val // 2, w // 2 + min_val // 2, h // 2 + min_val // 2))

        rgb_image = np.array(rgb_image.resize((self.img_size, self.img_size), Image.LANCZOS))

        lab_image = rgb2lab(rgb_image)
        ## Normalize to [0, 1]
        l_image = (np.clip(lab_image[:, :, 0:1], 0.0, 100.0) + 0.0) / (100.0 + 0.0)
        a_image = (np.clip(lab_image[:, :, 1:2], -86.0, 98.0) + 86.0) / (98.0  + 86.0)
        b_image = (np.clip(lab_image[:, :, 2:3], -107.0, 94.0) + 107.0) / (94.0 + 107.0)
        ab_image = np.concatenate((a_image, b_image), axis=2)


        l_image = torch.from_numpy(np.transpose(l_image, (2, 0, 1)).astype(np.float32)).repeat((3, 1, 1))
        if self.gray_only:
            return l_image

        ab_image = torch.from_numpy(np.transpose(ab_image, (2, 0, 1)).astype(np.float32))
        zero = torch.zeros((1, ab_image.shape[1], ab_image.shape[2]))
        ab_image = torch.cat([zero, ab_image], dim=0)
        if self.return_rgb:
            return l_image, ab_image, rgb_image_return
        else:
            return l_image, ab_image

    def get_img_path(self, index):
        path = self.paths[index]
        return path

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'ColorDataset'

