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


