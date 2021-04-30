import numpy as np 
import os
import math
import random
import cv2
from time import time

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.utils.data as Data

import ColorLog as debug


image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def persp_crop(img, corners, height, width):
    dst_points = np.array([(width, height), (0, height), (0, 0), (width, 0)], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
    dst = cv2.warpPerspective(img, transform_matrix, (width, height),flags=cv2.INTER_CUBIC)
#     dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
#     dst[:,:,0] = cv2.equalizeHist(dst[:,:,0])
#     dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
    return dst

def decode(preds):
    char_list = []
    code_list = []
    for i in range(len(preds)):
        if preds[i] != NUM_CHAR-1 and (not (i>0 and preds[i] == preds[i-1])):
            char_list.append(chars[preds[i]])
            code_list.append(preds[i])
    return code_list, char_list
    
def label_trans(label_list):
    assert len(label_list)==7
    out = [0]*7
    for ii, el in enumerate(label_list):
        if ii==0:
            out[ii] = int(el)
            if out[ii] == NUM_PROV-1:
                out[ii] = NUM_CHAR-1
        elif ii ==1:
            out[ii] = int(el)+NUM_PROV-1
            if out[ii] == NUM_ALPB-1:
                out[ii] = NUM_CHAR-1
        else:
            out[ii] = int(el)+NUM_PROV-1
            if out[ii] == NUM_ADS-1:
                out[ii] = NUM_CHAR-1
    return out


def list_images(basePath, contains=None):
    # return the set of files that are valid
    print(debug.INFO+"Loading data under %s"%basePath)
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath
                
class labelFpsDataLoader(Data.Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
#         plt.imshow(img[:,:,::-1])
#         plt.show()
        # img = img.astype('float32')
        lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        # fps = [[int(eel) for eel in el.split('&')] for el in iname[3].split('_')]
        # leftUp, rightDown = [min([fps[el][0] for el in range(4)]), min([fps[el][1] for el in range(4)])], [
        #     max([fps[el][0] for el in range(4)]), max([fps[el][1] for el in range(4)])]
        
#         print(debug.DEBUG,iname)
        
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]
        croppedImage = img[leftUp[1]:rightDown[1],leftUp[0]:rightDown[0]]
        resizedImage = cv2.resize(croppedImage, self.img_size)
#         cv2.imshow('plate',resizedImage)
#         cv2.waitKey(0)
#         print(resizedImage.shape)
        resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
#         plt.imshow(np.transpose(resizedImage, (1,2,0)))
#         plt.show()
        
#         cv2.imshow('plate',np.transpose(resizedImage, (1,2,0)))
#         cv2.waitKey(0)
        
        return resizedImage, new_labels, lbl, img_name, iname
             
class labelFpsPathDataLoader(Data.Dataset):
    def __init__(self, pathtxt, baseDir, imgSize, is_transform=None):
#         self.img_dir = img_dir
#         self.img_paths = []
#         for i in range(len(img_dir)):
#             self.img_paths += [el for el in list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        print(debug.INFO+"Loading data under %s"%pathtxt)
        f = open(pathtxt)
        self.img_paths = [os.path.join(baseDir, line.rstrip('\n')) for line in f.readlines()]
        f.close()
#         print("init")
#         print(self.img_paths)
        
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
#         img = cv2.imread(img_name)
        img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
#         img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # img = img.astype('float32')
        lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]
#         old_lbl = lbl.split('_')[:7]
#         print("lbl",len(old_lbl))
#         new_lbl = label_trans(lbl.split('_')[:7])
#         print([chars[x] for x in new_lbl])
        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        
#         plt.imshow(img[:,:,::-1])
#         plt.imshow(img)
#         plt.show()
#         input()
        # fps = [[int(eel) for eel in el.split('&')] for el in iname[3].split('_')]
        # leftUp, rightDown = [min([fps[el][0] for el in range(4)]), min([fps[el][1] for el in range(4)])], [
        #     max([fps[el][0] for el in range(4)]), max([fps[el][1] for el in range(4)])]
        
#         print(debug.DEBUG,iname)
        
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]
#         print(img.shape)
        croppedImage = img[leftUp[1]:rightDown[1],leftUp[0]:rightDown[0]]
#         print(croppedImage.shape)
        resizedImage = cv2.resize(croppedImage, self.img_size)
        resizedImage = np.expand_dims(resizedImage,0)
#         cv2.imshow('plate',resizedImage)
#         cv2.waitKey(0)
#         print(resizedImage.shape)
#         resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
#         plt.imshow(np.transpose(resizedImage, (1,2,0)))
#         plt.show()
        
#         cv2.imshow('plate',np.transpose(resizedImage, (1,2,0)))
#         cv2.waitKey(0)
        
        return resizedImage, new_labels, lbl, img_name, iname
      
class labelLoader(Data.Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
#         img = cv2.imread(img_name)
#         # img = img.astype('float32')
#         resizedImage = cv2.resize(img, self.img_size)
#         resizedImage = np.transpose(resizedImage, (2,0,1))
#         resizedImage = resizedImage.astype('float32')
#         resizedImage /= 255.0
        lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        # fps = [[int(eel) for eel in el.split('&')] for el in iname[3].split('_')]
        # leftUp, rightDown = [min([fps[el][0] for el in range(4)]), min([fps[el][1] for el in range(4)])], [
        #     max([fps[el][0] for el in range(4)]), max([fps[el][1] for el in range(4)])]
        
#         print(debug.DEBUG,iname)
        
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
#         ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
#         new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
#                       (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]

        return lbl, img_name, iname


