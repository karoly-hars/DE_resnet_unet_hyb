import math
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


HEIGHT = 256
WIDTH = 320

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def scale_and_crop_img(img):
    img = img[..., ::-1]
    # resizing
    scale = max(WIDTH/img.shape[1], HEIGHT/img.shape[0])
    img = cv2.resize(
        img, (math.ceil(img.shape[1]*scale), math.ceil(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST
    )
                                        
    # center crop to input size
    y_crop = img.shape[0] - HEIGHT
    x_crop = img.shape[1] - WIDTH
    img = img[math.floor(y_crop/2):img.shape[0]-math.ceil(y_crop/2),
              math.floor(x_crop/2):img.shape[1]-math.ceil(x_crop/2)]
    return img
    
    
def transform_img(img):   
    img = data_transform(img)
    img = img[None, :, :, :]
    return img


def load_img(img_path):
    img = cv2.imread(img_path)
    img = scale_and_crop_img(img)
    img = transform_img(img)
    return img
    

def pred_to_gray(pred):
    pred = np.transpose(pred, (1, 2, 0))
    max_dist = 10.0
    depth_img = pred
    depth_img[depth_img > max_dist] = max_dist
    depth_img = depth_img / max_dist
    
    depth_img = np.array(depth_img*255.0, dtype=np.uint8)
    depth_img = cv2.resize(depth_img, (WIDTH, HEIGHT))
    bgr_depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
    bgr_depth_img = np.clip(bgr_depth_img, 0, 255)
    return bgr_depth_img 


def show_img_pred(img, pred):
    plt.figure()
    plt.subplot(1, 2, 1)
    img = correct_img(img)
    plt.imshow(img)
    
    plt.subplot(1, 2, 2)
    pred = np.transpose(pred, (1, 2, 0))
    plt.imshow(pred[:, :, 0])
    plt.show()
    

def correct_img(img):
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (std * img + mean)
    img = np.clip(img, 0, 1)
    return img
