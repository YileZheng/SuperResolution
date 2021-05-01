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

from load_data import labelFpsPathDataLoader, label_trans, decode


def eval(model, test_tar, bsz, img_size, perspect=False):
    use_gpu = True
    count, error, correct = 0, 0, 0
    dst = labelFpsPathDataLoader(test_tar,"CCPD2019", img_size, is_transform=perspect)
#     dst = labelFpsDataLoader(test_tar, img_size)
    
    testloader = Data.DataLoader(dst, batch_size=bsz, shuffle=True, num_workers=0)
    start = time()
#     corrs_eachchar = np.zeros((7))
    print(debug.INFO+"start inference..")
    corrs_eachinst =[]
    for i, (XI,_, labels, ims, _) in enumerate(testloader):
        
        corr_eachinst =[]
        count += len(labels)
        YI = np.array([label_trans(el.split('_')[:7]) for el in labels])
        if use_gpu:
            x = Variable(torch.Tensor(XI).cuda())
            lbl = Variable(torch.LongTensor(YI).cuda())
        else:
            x = Variable(torch.Tensor(XI))
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



