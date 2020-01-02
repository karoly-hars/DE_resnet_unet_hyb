import cv2
import numpy as np
import torch
import argparse
import time
from utils import transform_img, pred_to_gray, scale_and_crop_img
from network import ResnetUnetHybrid


def run_vid(model, input_path, use_gpu):
    # capture and run video
    print('Inferencing video frames...')
    start = time.time()
    capture = cv2.VideoCapture(input_path)
    frame_cnt = 0
    
    if not capture.isOpened():
        print('ERROR: Failed to open video.')
        return

    while True:
        success, frame = capture.read()
        # stop when finished, or when interrupted by the user
        if not success:
            print('Finished.')
            break
            
        if cv2.waitKey(1) == ord('q'):
            print('Interrupted by user.')
            break
        
        frame_cnt += 1  # count frames for later report
        frame = scale_and_crop_img(frame)
        img = transform_img(frame)
        img = torch.Tensor(img)
            
        if use_gpu:
            img = img.cuda()
            
        pred = model(img)
        pred = pred.cpu()[0].data.numpy()
          
        pred = pred_to_gray(pred)
                      
        conc = np.concatenate((frame[..., ::-1], pred), axis=1)
        cv2.imshow('video', conc)

    end = time.time()
    print('\n{} frames evaluated in {}s'.format(int(frame_cnt), round(end-start, 3)))
    print('{:.2f} FPS'.format(frame_cnt/(end-start)))

    capture.release()
    cv2.destroyAllWindows()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='path to the input video')
    return parser.parse_args()


def main():
    args = get_arguments()
    
    # switching to GPU if possible
    use_gpu = torch.cuda.is_available()
    print('Using GPU:', use_gpu)    
    
    # loading model
    print('Loading model...')
    model = ResnetUnetHybrid.load_pretrained(use_gpu=use_gpu)
            
    # setting model to evaluation mode
    model.eval()
    
    run_vid(model, args.input_path, use_gpu)
    

if __name__ == '__main__':
    main()
