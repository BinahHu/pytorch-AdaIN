import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data

from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from sampler import InfiniteSamplerWrapper

from net import Net, vgg, decoder
from datasets import ColorDataset, GrayDataset, train_transform
from util import adjust_learning_rate, zero_grad

cudnn.benchmark = True






parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--aest_dir', type=str, required=True,
                    help='Directory path to a batch of Aesthetic images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--ias', type=str, default='models/ias_color.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--name', default='hope',
                    help='Name of this model')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--aest_weight', type=float, default=10.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(os.path.join(args.save_dir, args.name))
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(os.path.join(args.log_dir, args.name))
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))


vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = Net(vgg, decoder, args)


network.ias.load_state_dict(torch.load(args.ias))

network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()
aest_tf = train_transform()

content_dataset = GrayDataset(args.content_dir, content_tf)
style_dataset = GrayDataset(args.style_dir, style_tf)
aest_dataset = ColorDataset(args.aest_dir, aest_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
aest_iter = iter(data.DataLoader(
    aest_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(aest_dataset),
    num_workers=args.n_threads))

opt_dec = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)
opt_gen = torch.optim.Adam(network.generator.parameters(), lr=args.lr)
opt_dis = torch.optim.Adam(network.discriminator.parameters(), lr=args.lr)

opts = [opt_dec, opt_gen, opt_dis]

for i in tqdm(range(args.max_iter)):
    # S1: Adjust lr and prepare data
    adjust_learning_rate(opts, iteration_count=i, args=args)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    aest_images, real_l_images, real_ab_images = next(aest_iter)
    aest_images = aest_images.to(device)
    real_l_images = real_l_images.to(device)
    real_ab_images = real_ab_images.to(device)

    # S2: Train decoder
    loss_c, loss_s, fake_imgs = network(content_images, style_images, aest_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_dec = loss_c + loss_s

    zero_grad(opts)
    loss_dec.backward()
    opt_dec.step()

    # S3: Train discriminator
    color_feat = network.get_color_feat(aest_images)

    # S4: Train generator

    # S5: Summary loss and save models

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_dec', loss_dec.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth.tar'.format(i + 1))
writer.close()
