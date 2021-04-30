"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

class Bicubic_Upsampler(nn.Module):
    def __init__(self):
        super(Bicubic_Upsampler, self).__init__()

        self.up = nn.Upsample(scale_factor=4, mode="bicubic")

    def forward(self, x):
        out = self.up(x)
        return out

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    os.makedirs("images/inference", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="cropped_base_16_48", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=64, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=192, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_shape = (opt.hr_height, opt.hr_width)

    # # Initialize generator
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    # upsampler = Bicubic_Upsampler()

    # # Set to inference mode
    generator.eval()
    # upsampler.eval()

    generator.load_state_dict(torch.load("saved_models/generator_0_20000.pth"))

    print("Yes_____-------------")

    # ----------
    # Inference
    # ----------
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    lr_transform = transforms.Compose(
        [
            transforms.Resize((opt.hr_height // 4, opt.hr_width // 4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataloader = os.listdir("..\\..\\CCPD2019\\%s" % opt.dataset_name)

    for i in range(len(dataloader)):

        img = Image.open("..\\..\\CCPD2019\\%s\\%s" % (opt.dataset_name, dataloader[i]))
        img_lr = lr_transform(img)
        img_lr = torch.reshape(img_lr, (1, img_lr.shape[0], img_lr.shape[1], img_lr.shape[2])).to(device)

        gen_hr = generator(img_lr)
        
        save_image(denormalize(gen_hr), "images/inference/{}.png".format(dataloader[i]), nrow=1, normalize=False)
        exit()
