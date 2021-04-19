import cv2
import torch
import torch.nn as nn
from torch.utils.data import *
from imutils import paths
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import os
import argparse
from time import time
from torch.optim import lr_scheduler
from utils.dsetparser import parse_dset_config
from utils.rpnet_loader import RPNetDLoader
from models.rpnet import wR2
from utils.utils import *

USE_WANDB = False
if USE_WANDB:
    import wandb
    wandb.init(project='alpr', entity='afzal', name='detection_skeleton_0')

def train_model(model, criterion, optimizer, lrScheduler, trainloader, evalloader, batchSize, num_epochs=300):
    for epoch in range(1, num_epochs):
        model.train()
        print(f'Starting Epoch {epoch}')
        lossAver = []
        lrScheduler.step()
        start = time()
        dset_len = len(trainloader)
        correct_pred = 0
        for i, (XI, YI) in enumerate(trainloader):
            # print('%s/%s %s' % (i, times, time()-start))
            YI = np.array([el.numpy() for el in YI]).T
            x = Variable(XI.cuda(0))
            y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # Compute and print loss
            loss = 0.0
            if len(y_pred) == batchSize:
                loss += 0.8 * nn.L1Loss().cuda()(y_pred[:][:2], y[:][:2]) #Penalizing more on box center coordinates
                loss += 0.2 * nn.L1Loss().cuda()(y_pred[:][2:], y[:][2:])
                lossAver.append(loss.item())

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_iou = IoU(YI*480, 480*y_pred.cpu().detach().numpy())
                bin_correct = (batch_iou >= 0.7)
                correct_pred += np.sum(bin_correct)

            if (i*batchSize)%4999 == 1:
                #old_img = retrieve_img(XI[0])
                #old_img = draw_bbox(old_img, YI[0],'g')
                #old_img = draw_bbox(old_img, y_pred[0],'r')
                #iou_val = IoU(YI[0:1]*480, 480*y_pred[0:1].cpu().detach().numpy())
                #write_image(old_img, str(iou_val[0])+'.jpg')
                torch.save(model.state_dict(), 'trained_models/checkpoints/save_ckpt.pth')
                if USE_WANDB:
                #    wandb.log({'Sample Detections': [wandb.Image(channels_last(old_img*255.0), caption='iou='+str(iou_val)+'.jpg')]})
                    wandb.save('trained_models/checkpoints/save_ckpt.pth')

            if i % 500 == 0:
                curloss = sum(lossAver)/len(lossAver)
                curacc = correct_pred/((i+1)*batchSize)
                if USE_WANDB:
                    wandb.log({'Current Training Loss':curloss}, step=int((epoch-1)*dset_len/500+i/500))
                    wandb.log({'Current Training Acc':curacc}, step=int((epoch-1)*dset_len/500+i/500))
                print('Epoch {}, Processed: {}, Time: {}, Loss: {}, Acc: {}'.format(epoch, i*batchSize, time()-start, curloss, curacc))
        print ('Epoch %s Trained, Loss: %s, Accuracy: %s, Time: %s\n' % (epoch, sum(lossAver) / len(lossAver), correct_pred/(dset_len*batchSize), time()-start))
        if USE_WANDB:
            wandb.log({'Epoch Train Loss': sum(lossAver)/len(lossAver)}, step = epoch)
            wandb.log({'Epoch Train Accuracy': correct_pred/(dset_len*batchSize)}, step = epoch)
            wandb.save('trained_models/epochs/save_epoch' + str(epoch) + '.pth')
        torch.save(model.state_dict(), 'trained_models/epochs/save_epoch' + str(epoch) + '.pth')
        #Begin Eval here
        val_model(model, evalloader, batchSize, epoch) 
    return 

def val_model(model, evalloader, batchSize, epoch):
    model.eval()
    lossAver = []
    start = time()
    dset_len = len(evalloader)
    correct_pred = 0
    for i, (XI, YI) in enumerate(evalloader):
        YI = np.array([el.numpy() for el in YI]).T
        x = Variable(XI.cuda(0))
        y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = 0.0
        loss += 0.8 * nn.L1Loss().cuda()(y_pred[:][:2], y[:][:2]) #Penalizing more on box center coordinates
        loss += 0.2 * nn.L1Loss().cuda()(y_pred[:][2:], y[:][2:])
        lossAver.append(loss.item())

        batch_iou = IoU(YI*480, 480*y_pred.cpu().detach().numpy())
        #print('batch_iou:')
        #print(batch_iou)
        bin_correct = (batch_iou >= 0.7)
        #print('Correct/Incorrect:')
        #print(bin_correct)
        correct_pred += np.sum(bin_correct)
        #print('Number correct so far: {}'.format(correct_pred))

        if (i*batchSize)%10000 == 0:
            #write_image(old_img, str(iou_val[0])+'.jpg')
            #print(iou_val)
            if USE_WANDB:
                old_img = retrieve_img(XI[0])
                old_img = draw_bbox(old_img, YI[0],'g')
                old_img = draw_bbox(old_img, y_pred[0],'r')
                iou_val = batch_iou[0] 
                wandb.log({'Sample Test Detections': [wandb.Image(channels_last(old_img*255.0), caption='iou='+str(iou_val)+'.jpg')]}, commit=False) #step=int((epoch-1)*dset_len*batchSize/10000 + i*batchSize/10000))

        if i % 500 == 0:
            curloss = sum(lossAver)/len(lossAver)
            print('Evaluating Epoch: {}, Processed: {}, Time: {}, Loss: {}, Accuracy: {}'.format(epoch, i*batchSize, time()-start, curloss, correct_pred/((i+1)*batchSize)))
    print ('Epoch %s Evaluated, Loss: %s, Accuracy: %s, Time Elapsed: %s\n' % (epoch, sum(lossAver) / len(lossAver), correct_pred/((i+1)*batchSize), time()-start))
    if USE_WANDB:
        wandb.log({'Eval Epoch Loss': sum(lossAver)/len(lossAver)}, step=epoch)
        wandb.log({'Eval Epoch Accuracy': correct_pred/((i+1)*batchSize)}, step=epoch)
    return

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dsetconf", default='dset_config/config.ccpd',
                help="path to the dataset config file")
ap.add_argument("-n", "--epochs", default=300,
                help="epochs for train")
ap.add_argument("-b", "--batchsize", default=5,
                help="batch size for train")
args = vars(ap.parse_args())

def main():
    numClasses = 4
    imgSize = (480, 480)
    origSize = (720, 1160)
    batchSize = args['batchsize']
    epochs = args['epochs']
    lr = 0.001
    momentum = 0.9

    if USE_WANDB:
        config = wandb.config
        config.imgSize = imgSize
        config.batchSize = batchSize
        config.epochs = epochs
        config.lr = lr
        config.momentum = momentum

    model = wR2(numClasses)
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())) # This piece of shit hangs 
            #the node pretty fucking badly, to the point that the script process is unkillable and have to restart 
            #the node to restore operation, which results in a stopped docker container removing all its contents. 
            #(https://github.com/pytorch/pytorch/issues/24081#issuecomment-557074611). Cant disable IOMMU in BIOS 
            #since working on a remote node. Got no choice but to work with a single GPU. 
            #In summary, as Linus Torvalds would say: FUCK YOU NVIDIA
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lrScheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    dset_conf = parse_dset_config(args['dsetconf'])
    #Loading Train Split
    trainloc=dset_conf['train']
    dst = RPNetDLoader(trainloc, imgSize)
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=1)
    #Loading Validation Split
    valloc=dset_conf['val']
    valdst = RPNetDLoader(valloc, imgSize)
    evalloader = DataLoader(valdst, batch_size=batchSize, shuffle=False, num_workers=1)
    print('Starting Training...')
    model_conv = train_model(model, criterion, optimizer, lrScheduler, trainloader, evalloader, batchSize, num_epochs=epochs)

if __name__=='__main__':
    main()

#image_loc = ['/media/largeHDD/afzal/dl_project/placeholder_data/']
#loader = ChaLocDataLoader(image_loc, img_size)
#item, label = loader.__getitem__(0)
#draw_bbox(item, label)
