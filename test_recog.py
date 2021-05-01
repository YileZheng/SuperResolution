import numpy as np 
import pandas as pd
import os
import math
import random
import cv2
from time import time

import matplotlib.pyplot as plt
import quandl
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
import wandb

import ColorLog as debug

from load_data import labelFpsPathDataLoader
from models import CRNN
from train import eval


wandb.init(project='plateRecog_crnn', entity='leleleooonnn')
config = wandb.config

# DEVICE = "cuda"
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
chars = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

NUM_PROV = len(provinces)
NUM_ALPB = len(alphabets)
NUM_ADS = len(ads)
NUM_CHAR = len(chars)
MODEL_PATH = "weights/recog/eager-vortex/crnn_192_64.pth14"

TRAINDIR = ['CCPD2019/train']
TESTDIR = ['CCPD2019/test']
VALDIR = ['CCPD2019/val']
VALTXT = "CCPD2019/splits/val.txt"
TRAINTXT = "CCPD2019/splits/train.txt"
TESTTXT = "CCPD2019/splits/test.txt"

LR = 0.001
PERSPECTIVE = True
PLATESIZE = (192,64) #(100,32)#(256,96)
BATCHSIZE = 1
EPOCH = 100



# data loading
# image size 720x1160x3
config.input_size = PLATESIZE
config.perspectiveTrans = PERSPECTIVE


# test on val dataset
torch.cuda.set_device(0)
torch.cuda.empty_cache()

# wandb.init(project='DigitRecog',settings=wandb.Settings(start_method="fork"))
# PLATESIZE = (256,96)
print("loading model...")
model = CRNN(imgH=PLATESIZE[1], nc=1, nclass=NUM_CHAR, nh=256, n_rnn=2, leakyRelu=False)
print("load model")
model.load_state_dict(torch.load(MODEL_PATH))
model.cuda()
model.eval()

print("evaluating...")
count, corrs_inst, precision, avgTime = eval(model=model, test_tar=VALTXT, img_size=PLATESIZE, bsz=BATCHSIZE, perspect=PERSPECTIVE)
print('************* Validation: total %s precision %s avgTime %s\n' % (count, precision, avgTime))
print('accu_all_corr',len(corrs_inst[corrs_inst==7]))
print('accu_perct',len(corrs_inst[corrs_inst==7])/count)
