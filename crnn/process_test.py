import os
import cv2
import math
import numpy as np
import torch
from PIL import Image
from torch import nn
import torchvision.transforms as transforms

from models_esrgan import *
from datasets import *

HEIGHT = 16
WIDTH = 48
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

global lr_transform
lr_transform = transforms.Compose(
    [
        transforms.Resize((HEIGHT, WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator_upconv = GeneratorRRDB(channels=3, filters=64, num_res_blocks=23).to(device)

generator_upconv.load_state_dict(torch.load("../esrgan/saved_models/inference/generator_upconv.pth"))

generator_upconv.eval()

generator_upsamp = Generator_Anti_Artifact(channels=3, filters=64, num_res_blocks=23).to(device)

generator_upsamp.load_state_dict(torch.load("../esrgan/saved_models/inference/generator_upsamp.pth"))

generator_upsamp.eval()

generator_upsamp_nn = Generator_Anti_Artifact_Nearest_Neighbor(channels=3, filters=64, num_res_blocks=23).to(device)

generator_upsamp_nn.load_state_dict(torch.load("../esrgan/saved_models/inference/generator_upsamp_nn.pth"))

generator_upsamp_nn.eval()

def get_1_LR_upsampled_nearest(img, box, height, width):

    # 1. cropping based on bounding box, then map to LR size (16 * 48)
    dst_points = np.array([(width, height), (0, height), (0, 0), (width, 0)], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(np.array([box[1], (box[0][0], box[1][1]), box[0], (box[1][0], box[0][1])],np.float32), dst_points)
    dst = cv2.warpPerspective(img, transform_matrix, (width, height), flags=cv2.INTER_CUBIC)
    # 2. histogram equalization
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
    dst[:,:,0] = cv2.equalizeHist(dst[:,:,0])
    dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
    # 3. upsampling 4x using nearest (e.g., 2x means upsampling [1, 2, 3] to [1, 1, 2, 2, 3, 3])
    dst = np.transpose(dst, (2, 0, 1))
    img_16_48 = torch.from_numpy(dst).unsqueeze(0)
    img_upsampled = nn.functional.interpolate(img_16_48, scale_factor=4)[0]
    # 4. convert to gray image
    img_upsampled = np.transpose(img_upsampled.numpy(), (1, 2, 0))
    img_upsampled = cv2.cvtColor(img_upsampled, cv2.COLOR_BGR2GRAY)
    return img_upsampled

def get_2_HR_box_upconv(img, box, height, width):

    # 1. cropping based on bounding box, then map to bounding box size (box[1][0] - box[0][0] * box[1][1] - box[0][1])
    dst_points = np.array([(box[1][0] - box[0][0], box[1][1] - box[0][1]), (0, box[1][1] - box[0][1]), (0, 0), (box[1][0] - box[0][0], 0)], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(np.array([box[1], (box[0][0], box[1][1]), box[0], (box[1][0], box[0][1])],np.float32), dst_points)
    dst = cv2.warpPerspective(img, transform_matrix, (box[1][0] - box[0][0], box[1][1] - box[0][1]), flags=cv2.INTER_CUBIC)
    # 2. use CCPD's inherent `low-res transform` function, to get a (16 * 48) image
    dst = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    img_16_48 = lr_transform(dst)
    img_16_48 = torch.reshape(img_16_48, (1, img_16_48.shape[0], img_16_48.shape[1], img_16_48.shape[2])).to(device)
    # 3. super-resolution inference, using upconv here
    img_hr = generator_upconv(img_16_48)
    img_hr = denormalize(img_hr).cpu().detach()[0]
    # 4. convert to gray image
    img_hr = np.transpose(img_hr.numpy()[::-1,:,:], (1, 2, 0))
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
    return img_hr

def get_3_HR_corner_upconv(img, corners, height, width):
    
    # 1. cropping based on corner points, then map to LR size (16 * 48)
    dst_points = np.array([(width, height), (0, height), (0, 0), (width, 0)], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
    dst = cv2.warpPerspective(img, transform_matrix, (width, height), flags=cv2.INTER_CUBIC)
    # 2. histogram equalization
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
    dst[:,:,0] = cv2.equalizeHist(dst[:,:,0])
    dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
    # 3. use CCPD's inherent `low-res transform` function
    dst = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    img_16_48 = lr_transform(dst)
    img_16_48 = torch.reshape(img_16_48, (1, img_16_48.shape[0], img_16_48.shape[1], img_16_48.shape[2])).to(device)
    # 4. super-resolution inference, using upconv here
    img_hr = generator_upconv(img_16_48)
    img_hr = denormalize(img_hr).cpu().detach()[0]
    # 5. convert to gray image
    img_hr = np.transpose(img_hr.numpy()[::-1,:,:], (1, 2, 0))
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
    return img_hr

def get_4_HR_box_upsamp(img, box, height, width):

    # 1. cropping based on bounding box, then map to bounding box size (box[1][0] - box[0][0] * box[1][1] - box[0][1])
    dst_points = np.array([(box[1][0] - box[0][0], box[1][1] - box[0][1]), (0, box[1][1] - box[0][1]), (0, 0), (box[1][0] - box[0][0], 0)], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(np.array([box[1], (box[0][0], box[1][1]), box[0], (box[1][0], box[0][1])],np.float32), dst_points)
    dst = cv2.warpPerspective(img, transform_matrix, (box[1][0] - box[0][0], box[1][1] - box[0][1]), flags=cv2.INTER_CUBIC)
    # 2. use CCPD's inherent `low-res transform` function, to get a (16 * 48) image
    dst = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    img_16_48 = lr_transform(dst)
    img_16_48 = torch.reshape(img_16_48, (1, img_16_48.shape[0], img_16_48.shape[1], img_16_48.shape[2])).to(device)
    # 3. super-resolution inference, using upconv here
    img_hr = generator_upsamp(img_16_48)
    img_hr = denormalize(img_hr).cpu().detach()[0]
    # 4. convert to gray image
    img_hr = np.transpose(img_hr.numpy()[::-1,:,:], (1, 2, 0))
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
    return img_hr

def get_5_HR_corner_upsamp(img, corners, height, width):
    
    # 1. cropping based on corner points, then map to LR size (16 * 48)
    dst_points = np.array([(width, height), (0, height), (0, 0), (width, 0)], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
    dst = cv2.warpPerspective(img, transform_matrix, (width, height), flags=cv2.INTER_CUBIC)
    # 2. histogram equalization
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
    dst[:,:,0] = cv2.equalizeHist(dst[:,:,0])
    dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
    # 3. use CCPD's inherent `low-res transform` function
    dst = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    img_16_48 = lr_transform(dst)
    img_16_48 = torch.reshape(img_16_48, (1, img_16_48.shape[0], img_16_48.shape[1], img_16_48.shape[2])).to(device)
    # 4. super-resolution inference, using upconv here
    img_hr = generator_upsamp(img_16_48)
    img_hr = denormalize(img_hr).cpu().detach()[0]
    # 5. convert to gray image
    img_hr = np.transpose(img_hr.numpy()[::-1,:,:], (1, 2, 0))
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
    return img_hr

def get_6_HR_box_upsamp_nn(img, box, height, width):

    # 1. cropping based on bounding box, then map to bounding box size (box[1][0] - box[0][0] * box[1][1] - box[0][1])
    dst_points = np.array([(box[1][0] - box[0][0], box[1][1] - box[0][1]), (0, box[1][1] - box[0][1]), (0, 0), (box[1][0] - box[0][0], 0)], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(np.array([box[1], (box[0][0], box[1][1]), box[0], (box[1][0], box[0][1])],np.float32), dst_points)
    dst = cv2.warpPerspective(img, transform_matrix, (box[1][0] - box[0][0], box[1][1] - box[0][1]), flags=cv2.INTER_CUBIC)
    # 2. use CCPD's inherent `low-res transform` function, to get a (16 * 48) image
    dst = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    img_16_48 = lr_transform(dst)
    img_16_48 = torch.reshape(img_16_48, (1, img_16_48.shape[0], img_16_48.shape[1], img_16_48.shape[2])).to(device)
    # 3. super-resolution inference, using upconv here
    img_hr = generator_upsamp_nn(img_16_48)
    img_hr = denormalize(img_hr).cpu().detach()[0]
    # 4. convert to gray image
    img_hr = np.transpose(img_hr.numpy()[::-1,:,:], (1, 2, 0))
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
    return img_hr

def get_7_HR_corner_upsamp_nn(img, corners, height, width):
    
    # 1. cropping based on corner points, then map to LR size (16 * 48)
    dst_points = np.array([(width, height), (0, height), (0, 0), (width, 0)], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
    dst = cv2.warpPerspective(img, transform_matrix, (width, height), flags=cv2.INTER_CUBIC)
    # 2. histogram equalization
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
    dst[:,:,0] = cv2.equalizeHist(dst[:,:,0])
    dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
    # 3. use CCPD's inherent `low-res transform` function
    dst = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    img_16_48 = lr_transform(dst)
    img_16_48 = torch.reshape(img_16_48, (1, img_16_48.shape[0], img_16_48.shape[1], img_16_48.shape[2])).to(device)
    # 4. super-resolution inference, using upconv here
    img_hr = generator_upsamp_nn(img_16_48)
    img_hr = denormalize(img_hr).cpu().detach()[0]
    # 5. convert to gray image
    img_hr = np.transpose(img_hr.numpy()[::-1,:,:], (1, 2, 0))
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
    return img_hr


