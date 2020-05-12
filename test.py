import os

import torch

from tqdm import tqdm

from PIL import Image
from pathlib import Path

from net import Net
from datasets import ColorDataset
from util import res_lab2rgb
import shutil

def test(args):

    # Device and output dir
    device = torch.device('cuda')
    out_dir = os.path.join(args.out_root, args.name)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    ref_dir = os.path.join(out_dir, "ref")
    Path(ref_dir).mkdir(exist_ok=True, parents=True)

    # Prepare datasets

    content_dataset = ColorDataset(args.content_dir, args.img_size)
    style_dataset = ColorDataset(args.style_dir, args.img_size, gray_only=True)
    aest_dataset = ColorDataset(args.aest_dir, args.img_size)
    LC = len(content_dataset)
    LS = len(style_dataset)
    LA = len(aest_dataset)

    # Prepare network

    network = Net(args)
    network.load_state_dict(torch.load(args.network))
    network.eval()
    network.to(device)

    # Save ref img
    for i in range(LC):
        path = content_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "content_{}.jpg".format(i)))
    for i in range(LS):
        path = style_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "style_{}.jpg".format(i)))
    for i in range(LA):
        path = aest_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "aest_{}.jpg".format(i)))

    # Start Test
    N = LC * LS * LA
    print("LC = {}, LS = {}, LA = {}, total output num = {}".format(LC, LS, LA, N))
    with tqdm(total=N) as t:
        with torch.no_grad():
            for i in range(LC):
                for j in range(LS):
                    for k in range(LA):
                        # S1: Prepare data and forward

                        content_l, content_ab = [x.to(device).unsqueeze(0) for x in content_dataset.__getitem__(i)]
                        style_l = style_dataset.__getitem__(j).to(device).unsqueeze(0)
                        aest_l, aest_ab = [x.to(device).unsqueeze(0) for x in aest_dataset.__getitem__(k)]
                        l_pred, ab_pred = network(content_l, content_ab, style_l, aest_ab)

                        # S2: Save
                        rgb_img, l_img = res_lab2rgb(l_pred.squeeze(0), ab_pred.squeeze(0))

                        img = Image.fromarray(rgb_img)
                        name = 'c{}_s{}_a{}_result.png'.format(i, j, k)
                        img.save(os.path.join(out_dir, name))

                        img = Image.fromarray(l_img, 'L')
                        name = 'c{}_s{}_gray.png'.format(i, j, k)
                        img.save(os.path.join(out_dir, name))

                        t.update(1)