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

USE_WANDB = False
if USE_WANDB:
    import wandb
    wandb.init(project='alpr', entity='afzal', name='detection_skeleton_0')

#Courtesy of https://github.com/detectRecog/CCPD/
class ChaLocDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        with open(img_dir, 'r') as file:
            self.img_paths = file.readlines()

        #self.img_dir = img_dir
        #self.img_paths = []
        #for i in range(len(img_dir)):
        #    self.img_paths += [el for el in paths.list_images(img_dir[i])]
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img_name = os.path.join('CCPD2019', img_name)
        img = cv2.imread(img_name.strip())
        if img is None:
            print('Could\'nt read image at path {}'.format(img_name))
            print('Later, Loser...')
            exit(0)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        #---Uncomment to generate image with bounding box
        #cv2.rectangle(img, (leftUp[0], leftUp[1]), (rightDown[0], rightDown[1]), (255, 0, 0), 1) 
        #cv2.imwrite('res1.jpg', img)
        #---

        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        assert img.shape[0] == 1160
        new_labels = [(leftUp[0] + rightDown[0])/(2*ori_w), (leftUp[1] + rightDown[1])/(2*ori_h), (rightDown[0]-leftUp[0])/ori_w, (rightDown[1]-leftUp[1])/ori_h]

        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0

        return resizedImage, new_labels


class wR2(nn.Module):
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x

def draw_bbox(image, bbox, color):  #Input/Output in (C, H, W), 0-1 range
    image_cp = np.copy(image)
    h, w = image_cp.shape[1], image_cp.shape[2]
    image_cp = np.float32(np.reshape(image_cp, (image_cp.shape[1],image_cp.shape[2], image_cp.shape[0])))
    topleft_x = int(bbox[0]*w - bbox[2]*w/2.)
    topleft_y = int(bbox[1]*h - bbox[3]*h/2.)
    botright_x = int(bbox[0]*w + bbox[2]*w/2.)
    botright_y = int(bbox[1]*h + bbox[3]*h/2.)
    if color == 'g':
        cl = (0, 255, 0)
    elif color == 'r':
        cl = (0, 0, 255)
    else:
        cl = (0, 0, 0)
    cv2.rectangle(image_cp, (topleft_x, topleft_y), (botright_x, botright_y), cl, 1) 
    return np.reshape(image_cp, (image_cp.shape[2], image_cp.shape[0], image_cp.shape[1]))

def write_image(image, filename): #Input in (C, H, W), 0-1 range
    image_cp = np.copy(image)
    image_cp = np.float32(np.reshape(image_cp, (image_cp.shape[1],image_cp.shape[2], image_cp.shape[0])))
    print('Writing Image...')
    cv2.imwrite(filename, image_cp*255.0)
    return

def retrieve_img(image):
    cp = np.copy(image)
    cp = np.reshape(cp, (cp.shape[1], cp.shape[2], cp.shape[0]))
    resized = cv2.resize(cp, (720, 1160))
    cp = np.reshape(resized, (resized.shape[2], resized.shape[0], resized.shape[1]))
    return cp

def channels_last(image):
    return np.reshape(image, (image.shape[1], image.shape[2], image.shape[0]))

def train_model(model, criterion, optimizer, lrScheduler, trainloader, evalloader, batchSize, num_epochs=25):
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

            if i % 10 == 0:
                curloss = sum(lossAver)/len(lossAver)
                curacc = correct_pred/(i*batchSize)
                if USE_WANDB:
                    wandb.log({'Current Training Loss':curloss, 'Current Training Acc':curacc}, step=int((epoch-1)*dset_len/10+i/10))
                print('Epoch {}, Processed: {}, Time: {}, Loss: {}, Acc: {}'.format(epoch, i*batchSize, time()-start, curloss, curacc))
        print ('Epoch %s Trained, Loss: %s, Accuracy: %s, Time: %s\n' % (epoch, sum(lossAver) / len(lossAver), correct_pred/(dset_len*batchSize), time()-start))
        if USE_WANDB:
            wandb.log({'Epoch Train Loss': sum(lossAver)/len(lossAver), 'Epoch Train Accuracy': correct_pred/(dset_len*batchSize)}, step = epoch)
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

        if (i*batchSize)%500 == 0:
            #write_image(old_img, str(iou_val[0])+'.jpg')
            #print(iou_val)
            if USE_WANDB:
                old_img = retrieve_img(XI[0])
                old_img = draw_bbox(old_img, YI[0],'g')
                old_img = draw_bbox(old_img, y_pred[0],'r')
                iou_val = batch_iou[0] 
                wandb.log({'Sample Test Detections': [wandb.Image(channels_last(old_img*255.0), caption='iou='+str(iou_val)+'.jpg')]}, step=epoch)

        if i % 10 == 0:
            curloss = sum(lossAver)/len(lossAver)
            print('Evaluating Epoch: {}, Processed: {}, Time: {}, Loss: {}, Accuracy: {}'.format(epoch, i*batchSize, time()-start, curloss, correct_pred/(i*batchSize)))
    print ('Epoch %s Evaluated, Loss: %s, Accuracy: %s, Time Elapsed: %s\n' % (epoch, sum(lossAver) / len(lossAver), correct_pred/(i*batchSize), time()-start))
    if USE_WANDB:
        wandb.log({'Eval Epoch Loss': sum(lossAver)/len(lossAver), 'Eval Epoch Accuracy': correct_pred/(i*batchSize)}, step=epoch)
    return

def IoU(boxa, boxb):
    boxA = np.zeros(boxa.shape)
    boxB = np.zeros(boxb.shape)
    boxA[:,:2] = boxa[:,:2] - boxa[:,2:]/2.
    boxA[:,2:] = boxa[:,:2] + boxa[:,2:]/2.
    boxB[:,:2] = boxb[:,:2] - boxb[:,2:]/2.
    boxB[:,2:] = boxb[:,:2] + boxb[:,2:]/2.

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[:,0], boxB[:,0])
    yA = np.maximum(boxA[:,1], boxB[:,1])
    xB = np.minimum(boxA[:,2], boxB[:,2])
    yB = np.minimum(boxA[:,3], boxB[:,3])

    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[:,2] - boxA[:,0] + 1) * (boxA[:,3] - boxA[:,1] + 1)
    boxBArea = (boxB[:,2] - boxB[:,0] + 1) * (boxB[:,3] - boxB[:,1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = np.divide(interArea, boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dsetconf", default='dset_config/config.ccpd',
                help="path to the dataset config file")
ap.add_argument("-n", "--epochs", default=25,
                help="epochs for train")
ap.add_argument("-b", "--batchsize", default=5,
                help="batch size for train")
ap.add_argument("-w", "--writeFile", default='wR2.out',
                help="file for output")
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
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())) # This piece of shit hangs the node. Fucker
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lrScheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    dset_conf = parse_dset_config(args['dsetconf'])
    #Loading Train Split
    trainloc=dset_conf['train']
    dst = ChaLocDataLoader(trainloc, imgSize)
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)
    #Loading Validation Split
    valloc=dset_conf['val']
    valdst = ChaLocDataLoader(valloc, imgSize)
    evalloader = DataLoader(valdst, batch_size=batchSize, shuffle=False, num_workers=4)
    print('Starting Training...')
    model_conv = train_model(model, criterion, optimizer, lrScheduler, trainloader, evalloader, batchSize, num_epochs=epochs)

if __name__=='__main__':
    main()

#image_loc = ['/media/largeHDD/afzal/dl_project/placeholder_data/']
#loader = ChaLocDataLoader(image_loc, img_size)
#item, label = loader.__getitem__(0)
#draw_bbox(item, label)
