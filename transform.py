import os
import cv2
import numpy as np


def getImageVar(img):
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imgVar

def persp_crop(img):
    dst_points = np.array([(440, 140), (0, 140), (0, 0), (440, 0)], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
    dst = cv2.warpPerspective(img, transform_matrix, (440, 140),flags=cv2.INTER_CUBIC)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
    dst[:,:,0] = cv2.equalizeHist(dst[:,:,0])
    dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
    return dst


if __name__ == "__main__":
    data_dir = "./CCPD2019/ccpd_base/"
    all_files = os.listdir(data_dir)

    count = 0
    max_total_light = 140*440*255
    ratio_list = []

    for i in all_files:

        corners = np.array(eval("[(" + i.split("-")[3].replace("&", ",").replace("_", "),(") + ")]"), np.float32) # [0] BR, [1] BL, [2] TL, [3] TR
        # print(corners)
        img = cv2.imread(data_dir + i)
        dst = persp_crop(img)
        cv2.imshow("image", dst)
        cv2.waitKey(0)

'''
    for i in all_files:
        # print(i)
        corners = np.array(eval("[(" + i.split("-")[3].replace("&", ",").replace("_", "),(") + ")]"), np.float32) # [0] BR, [1] BL, [2] TL, [3] TR
        # print(corners)
        img = cv2.imread(data_dir + i)
        # # print(img.shape) # (1160, 720, 3)
        
        dst_points = np.array([(440, 140), (0, 140), (0, 0), (440, 0)], np.float32)
        transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
        l = [] 
        dst = cv2.warpPerspective(img, transform_matrix, (440, 140),flags=cv2.INTER_CUBIC)

        
        # print(getImageVar(dst))
        l.append(dst)

        # light_ratio = (cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY).sum()/max_total_light)
        # dst = dst * 0.5 / light_ratio
        # dst = np.clip(dst, 0, 255).astype(np.uint8)

        # l.append(dst)

        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) # [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
        # dst = cv2.filter2D(dst, -1, kernel = kernel)
        
        # l.append(dst)
        
        # b, g, r = cv2.split(dst)
        # b = cv2.equalizeHist(b)
        # g = cv2.equalizeHist(g)
        # r = cv2.equalizeHist(r)
        # dst = cv2.merge([b, g, r])
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
        dst[:,:,0] = cv2.equalizeHist(dst[:,:,0])
        dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
        l.append(dst)
        # dst 
        # print(type(dst))
        # print(dst)
        # print(dst.shape)
        

        dst = np.hstack(l)

        # # print(dst[0])
        cv2.imshow("image", dst)
        
        cv2.waitKey(0)
        # break
        count += 1
        if (count % 100 == 0):
            print(count)
'''