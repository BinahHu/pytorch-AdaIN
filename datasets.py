from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils import data
import numpy as np
from skimage.color import rgb2lab

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class GrayDataset(data.Dataset):
    def __init__(self, root, transform):
        super(GrayDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('L')
        img = self.transform(img)
        img = img.repeat(3, 1, 1)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'GrayDataset'


class ColorDataset(data.Dataset):
    def __init__(self, root, transform):
        super(ColorDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        rgb_image = np.array(img)

        lab_image = rgb2lab(rgb_image)
        l_image = np.transpose(lab_image[:, :, :1], (2, 0, 1)).astype(np.float32)
        ab_image = np.transpose(lab_image[:, :, 1:], (2, 0, 1)).astype(np.float32)

        img = self.transform(img)
        l_image = self.transform(l_image)
        ab_image = self.transform(ab_image)
        return img, l_image, ab_image

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'ColorDataset'

