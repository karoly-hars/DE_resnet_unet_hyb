import cv2
import numpy as np
import torch
from torch.autograd import Variable
from os.path import isfile, join
from os import listdir
from utils import data_transform
import net
    

def compute_errors(model, use_gpu):  
    
    # loading test images
    test_dir = 'NYU_depth_v2_test_set'
    
    test_img_paths = [join(test_dir,f) for f in listdir(test_dir) if isfile(join(test_dir, f)) and f.endswith('_rgb.png')]
    test_label_paths = [join(test_dir,f) for f in listdir(test_dir) if isfile(join(test_dir, f)) and f.endswith('_dpth.png')]        
    test_img_paths.sort()
    test_label_paths.sort()
    test_files = [[x,y] for (x,y) in zip(test_img_paths,test_label_paths)]
    
    
    # upsampler
    resizer = torch.nn.Upsample(size=(466, 582), mode='bilinear', align_corners=False)
    

    preds = np.zeros((466,582,654))
    labels = np.zeros((466,582,654))
    

    print('\nRunnning the model on the test set...')

    for idx,pair in enumerate(test_files):
        img = cv2.imread(pair[0])[...,::-1]
        label = cv2.imread(pair[1],-1)/6000 # this division is necessary, because the depth values were multiplied by 6000 and saved as uint16 images
        
        # resize and center crop image to input size
        img = cv2.resize(img, (int(img.shape[1]*0.55), int(img.shape[0]*0.55)), interpolation = cv2.INTER_NEAREST)
        img = img[4:img.shape[0]-4, 16:img.shape[1]-16]
          
        # center crop label to output size
        label = label[7:label.shape[0]-7, 29:label.shape[1]-29]
    
        # load into the labels array
        labels[:,:,idx] = label
    
        img = data_transform(img)
        img = img[None,:,:,:]
        
        # running model on the image
        if use_gpu:
            img = Variable(img.cuda())
        else:
            img = Variable(img)
            
        # running model and upsampling the prediction
        pred = model(img) 
        pred = resizer(pred)
        pred = pred.cpu().data.numpy()
     
        # load into the predictions array
        preds[:,:,idx] = pred[0,0,:,:]
        
    
    
    # calculating errors
    rel_error = np.mean(np.abs(preds - labels)/labels)
    print('\nMean Absolute Relative Error: {:.6f}'.format(rel_error))
    
    rmse = np.sqrt(np.mean((preds - labels)**2))
    print('Root Mean Squared Error: {:.6f}'.format(rmse))
    
    log10 = np.mean(np.abs(np.log10(preds) - np.log10(labels)))
    print('Mean Log10 Error: {:.6f}'.format(log10))
    
    acc = np.maximum(preds/labels, labels/preds)
    delta1 = np.mean(acc < 1.25)
    print('Delta1: {:.6f}'.format(delta1))
    
    delta2 = np.mean(acc < 1.25**2)
    print('Delta2: {:.6f}'.format(delta2))
    
    delta3 = np.mean(acc < 1.25**3)
    print('Delta3: {:.6f}'.format(delta3))    
    
    
    

def main():
    # switching to GPU if possible
    use_gpu = torch.cuda.is_available()
    print('\nusing GPU:', use_gpu)    

    
    # loading model
    print('\nLoading model...')
    model = net.hyb_net(use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
            
    # setting model to evalutation mode
    model.eval()
    
    compute_errors(model, use_gpu)
  


if __name__ == "__main__":
    main()

