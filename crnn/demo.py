from load_data import *
import cv2

UPSAMP_METHODS = ['LR', 'box_upsamp', 'box_upconv', 'box_upsamp_nn', 'cor_upconv', 'cor_upsamp', 'cor_upsamp_nn', 'origin']
VALTXT = "../../CCPD2019/splits/val.txt"
TRAINTXT = "../../CCPD2019/splits/train.txt"
TESTTXT = "../../CCPD2019/splits/test.txt"
method  = 'cor_upsamp_nn'
dst = labelFpsPathDataLoader(VALTXT,"../../CCPD2019", (192,64), is_transform=method)
print(dst[0][0][0])
