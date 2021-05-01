import numpy as np 
import pandas as pd
import os
import math
import random
import cv2
from time import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
import wandb

import ColorLog as debug

wandb.init(project='plateRecog_crnn', entity='leleleooonnn')
config = wandb.config

LR = 0.001
PERSPECTIVE = True
PLATESIZE = (192,64) #(100,32)#(256,96)
BATCHSIZE = 128
EPOCH = 100


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
chars = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
         "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", 'A',
         'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

NUM_PROV = len(provinces)
NUM_ALPB = len(alphabets)
NUM_ADS = len(ads)
NUM_CHAR = len(chars)


TRAINDIR = ['CCPD2019/train']
TESTDIR = ['CCPD2019/test']
VALDIR = ['CCPD2019/val']
VALTXT = "CCPD2019/splits/val.txt"
TRAINTXT = "CCPD2019/splits/train.txt"
TESTTXT = "CCPD2019/splits/test.txt"

# data loading
# image size 720x1160x3
config.input_size = PLATESIZE
config.perspectiveTrans = PERSPECTIVE

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def persp_crop(img, corners, height, width):
    dst_points = np.array([(width, height), (0, height), (0, 0), (width, 0)], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
    dst = cv2.warpPerspective(img, transform_matrix, (width, height),flags=cv2.INTER_CUBIC)
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
    #dst[:,:,0] = cv2.equalizeHist(dst[:,:,0])
    #dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
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
    def __init__(self, pathtxt, baseDir, imgSize, is_transform=False):
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
        img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
        lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]
#         old_lbl = lbl.split('_')[:7]
#         print("lbl",len(old_lbl))
#         new_lbl = label_trans(lbl.split('_')[:7])
#         print([chars[x] for x in new_lbl])
        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')

#         plt.imshow(img[:,:,::-1])
#         plt.imshow(img)
#         plt.show()

#         print(debug.DEBUG,iname)

        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]
#         print(img.shape)
        if not self.is_transform:
            croppedImage = img[leftUp[1]:rightDown[1],leftUp[0]:rightDown[0]]
            resizedImage = cv2.resize(croppedImage, self.img_size)
        else:
            corners = np.array([[int(eel) for eel in el.split('&')] for el in iname[3].split('_')], np.float32)
            resizedImage = persp_crop(img, corners, self.img_size[1], self.img_size[0])
#         print(croppedImage.shape)
        resizedImage = np.expand_dims(resizedImage,0)
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
#         plt.imshow(np.transpose(resizedImage, (1,2,0)))
#         plt.show()

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

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        # 1x64x192 1x32x100
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x96 64x16x50
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x16x48 128x8x25
        convRelu(2)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 128x8x24
        convRelu(3, True)
        convRelu(4)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x25 256x4x26
        convRelu(5, True)
        convRelu(6)
        cnn.add_module('pooling{0}'.format(4),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x26 512x2x27
        convRelu(7, True)  # 512x1x25 512x1x26

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


def eval(model, test_tar):
    use_gpu = True
    count, error, correct = 0, 0, 0
    dst = labelFpsPathDataLoader(test_tar,"CCPD2019", PLATESIZE, is_transform=PERSPECTIVE)
#     dst = labelFpsDataLoader(test_tar, PLATESIZE)
    bsz = BATCHSIZE
    testloader = Data.DataLoader(dst, batch_size=bsz, shuffle=True, num_workers=8)
    start = time()
#     corrs_eachchar = np.zeros((7))
    corrs_eachinst =[]
    for i, (XI,_, labels, ims, _) in enumerate(testloader):
        
        corr_eachinst =[]
        count += len(labels)
        YI = np.array([label_trans(el.split('_')[:7]) for el in labels])
        if use_gpu:
            x = Variable(XI.cuda())
            lbl = Variable(torch.LongTensor(YI).cuda())
        else:
            x = Variable(XI)
            lbl = Variable(torch.LongTensor(YI))

        y_pred = model(x)
#         print(y_pred.shape)
        
        _, preds = y_pred.max(2)
        preds = preds.transpose(1, 0).contiguous()
        for i in range(lbl.shape[0]):
            n_correct = 0
            sim_preds, _ = decode(preds[i].data)
#             print(sim_preds,lbl[i].data)
            for pred, target in zip(sim_preds, lbl[i].data):
                if pred == target:
                    n_correct += 1
            corr_eachinst.append(n_correct)
            

        corrs_eachinst = np.append(corrs_eachinst,corr_eachinst)
        
        
        if i%10 ==1:
            print(debug.INFO+"image: {}, inst:{}".format(count,np.mean(corrs_eachinst)))#, corrs_eachchar/count))
    wandb.log({'val':{
        'image#':count,
        'corr_in_instance':np.mean(corrs_eachinst),
        'accu_instance':np.mean(corrs_eachinst)/7,
        'accu_all_corr':len(corrs_eachinst[corrs_eachinst==7]),
        'corr_distrb':wandb.Histogram(corrs_eachinst),
        'corr_inst':corrs_eachinst
              }})  
    
    return count, corrs_eachinst, np.mean(corrs_eachinst)/7, (time()-start) / count


def train_model(model, trainloader, criterion, optimizer,batchSize, testDirs,storeName, num_epochs=25, logFile="./train_log.txt"):
    # since = time.time()
    use_gpu = True
    lrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, verbose=True)
    cnt = 0

    
    for epoch in range(num_epochs):
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time()
        print(debug.INFO+"Epoch {} started at {}".format(epoch,start))

        for i, (XI, _, labels, _, _) in enumerate(trainloader):
            cnt +=1
            if not len(XI) == batchSize:
                
                continue
                        
            YI = [label_trans(el.split('_')[:7]) for el in labels]
#             Y = np.array([el.numpy() for el in Y]).T
            if use_gpu:
                x = Variable(XI.cuda())
                lbl = Variable(torch.LongTensor(YI).cuda())
#                 y = Variable(torch.FloatTensor(Y).cuda(), requires_grad=False)
            else:
                x = Variable(XI)
                lbl = Variable(torch.LongTensor(YI))
    
#             print(debug.INFO+"input shape {}".format(x.shape))
            y_pred = model(x)
#             print(debug.INFO+"output size:",y_pred.shape)
#             print(debug.INFO+"output shape {}".format([yy.shape for yy in y_pred]))
#             try:
#                 y_pred = model(x)
#                 print(debug.INFO+"output shape {}".format(y_pred.shape))
                
#             except:
#                 print(debug.WARN+"iter %d model prediction fails"%i)
#                 continue
                
            # Compute and print loss
#             loss = 0.0
#             train_correct = []
#             loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:][:2], y[:][:2])
#             loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:][2:], y[:][2:])
            y_pred = F.log_softmax(y_pred,dim=2)
            preds_size = Variable(torch.IntTensor([y_pred.size(0)] * batchSize))
            tars_size = Variable(torch.IntTensor([7] * batchSize))
#             print("loss input",y_pred,lbl,preds_size,tars_size)
            loss  = criterion(y_pred,lbl,preds_size,tars_size)
#             for j in range(7):
#                 l = lbl[:,j]
#                 loss += criterion(y_pred[j], l)
#                 train_correct.append(np.argmax(y_pred[j],axis=1))
#                 acc = len(train_correct[train_correct==0])/len(train_correct)

#             def isEqual(labelGT, labelP):
#                 compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
#                 # print(sum(compare))
#                 return sum(compare)
            
#             for ii in range(batchSize):
#                 if isEqual(labelPred, YI[ii]) == 7:
#                     correct += 1


                    
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#             print(loss)
            lossAver.append(loss.item())

#             try:
#                 print(loss)
#                 lossAver.append(loss.data[0])
#             except:
#                 print(debug.ERR+"iter %d lossAver append error"%i)
            if cnt % 100 == 0:
                wandb.log({'train':{
                    'cur_loss':loss,
                    'ave_loss':np.mean(lossAver)
                          }})  
            
            if i % 50 == 1:
                print('trained %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver) / len(lossAver) if len(lossAver)>0 else 'NoLoss'))
                with open(logFile, 'a') as outF:
                    outF.write('trained %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver) / len(lossAver) if len(lossAver)>0 else 'NoLoss'))
                torch.save(model.state_dict(), storeName)
        print ('*************Epoch %s Avrg Training loss %s Elapsed %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        
        model.eval()
        count, correct, precision, avgTime = eval(model, testDirs)
        with open(logFile, 'a') as outF:
            outF.write('Epoch %s Avrg Training loss %s Elapsed %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
            outF.write('************* Validation: total %s precision %s avgTime %s\n' % (count, precision, avgTime))
        torch.save(model.state_dict(), storeName + str(epoch))
        print('************* Validation: total %s precision %s avgTime %s\n' % (count, precision, avgTime))
    return model

# training
torch.cuda.set_device(1)
torch.cuda.empty_cache()

# model = DigitRecog(PLATESIZE).cuda()
model = CRNN(imgH=32, nc=1, nclass=NUM_CHAR, nh=256, n_rnn=2, leakyRelu=False).cuda()
# criterion = nn.CrossEntropyLoss()
wandb.watch(model)
criterion = nn.CTCLoss(blank=NUM_CHAR-1,reduction='mean').cuda()
optimizer = optim.Adam(model.parameters(),lr=LR)
# lrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# optimizer_conv = optim.RMSprop(model_conv.parameters(), lr=0.01, momentum=0.9)
# optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print(debug.INFO+"Start loading dataset...")
# dst = labelFpsDataLoader(TRAINDIR, PLATESIZE)
dst = labelFpsPathDataLoader(TRAINTXT,"CCPD2019", PLATESIZE, is_transform=PERSPECTIVE)
print(debug.INFO+"Got dataset size %d"%len(dst))
trainloader = Data.DataLoader(dst, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
print(debug.INFO+"Done loading dataset")

print(debug.INFO+"Start training")
model = train_model(model=model, trainloader=trainloader, criterion=criterion, optimizer=optimizer,
            batchSize=BATCHSIZE, testDirs=VALTXT,storeName='./weight/crnn_192_64.pth', num_epochs=EPOCH, logFile="./train_log.txt")
