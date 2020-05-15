import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from sampler import InfiniteSamplerWrapper

from net import Net
from datasets import ColorDataset
from util import adjust_learning_rate

cudnn.benchmark = True

def train(args):

    # Device, save and log configuration

    device = torch.device('cuda')
    save_dir = Path(os.path.join(args.save_dir, args.name))
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(os.path.join(args.log_dir, args.name))
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Prepare datasets

    content_dataset = ColorDataset(args.content_dir, args.img_size)
    style_dataset = ColorDataset(args.style_dir, args.img_size, gray_only=True)
    aest_dataset = ColorDataset(args.aest_dir, args.img_size)

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

    # Prepare network

    network = Net(args)
    network.train()
    network.to(device)

    # Training options

    opt_L = torch.optim.Adam(network.L_path.parameters(), lr=args.lr)
    opt_AB = torch.optim.Adam(network.AB_path.parameters(), lr=args.lr)

    opts = [opt_L, opt_AB]

    # Start Training

    for i in tqdm(range(args.max_iter)):
        # S1: Adjust lr and prepare data

        adjust_learning_rate(opts, iteration_count=i, args=args)

        content_l, content_ab = [x.to(device) for x in next(content_iter)]
        style_l = next(style_iter).to(device)
        aest_l, aest_ab = [x.to(device) for x in next(aest_iter)]

        # S2: Forward

        l_pred, ab_pred = network(content_l, content_ab, style_l, aest_ab)

        # S3: Calculate loss

        loss_c, loss_s = network.c_s_loss(l_pred, content_l, style_l)
        loss_a = network.a_loss(ab_pred, aest_ab)

        loss_cw = args.content_weight * loss_c
        loss_sw = args.style_weight * loss_s
        loss_aw = args.aest_weight * loss_a

        loss = loss_cw + loss_sw + loss_aw

        # S4: Backward

        for opt in opts:
            opt.zero_grad()
        loss.backward()
        for opt in opts:
            opt.step()

        # S5: Summary loss and save subnets

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)
        writer.add_scalar('loss_aest', loss_a.item(), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = network.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'network_iter_{:d}.pth.tar'.format(i + 1))
    writer.close()
