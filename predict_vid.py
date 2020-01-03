import cv2
import numpy as np
import torch
import argparse
import time
import image_utils
from network import ResnetUnetHybrid


def run_vid(input_path):
    """Load, transform and inference the frames of a video. Display the predictions with the input frames."""
    # switch to GPU if possible
    use_gpu = torch.cuda.is_available()
    print('Using GPU:', use_gpu)

    # load model
    print('Loading model...')
    model = ResnetUnetHybrid.load_pretrained(use_gpu=use_gpu)

    # setting model to evaluation mode
    model.eval()

    # start running the video
    print('Inferencing video frames...')
    start = time.time()
    capture = cv2.VideoCapture(input_path)
    frame_cnt = 0
    
    if not capture.isOpened():
        print('ERROR: Failed to open video.')
        return -1

    while True:
        success, frame = capture.read()
        # stop when finished, or when interrupted by the user
        if not success:
            print('Finished.')
            break
        if cv2.waitKey(1) == ord('q'):
            print('Interrupted by user.')
            break
        
        frame_cnt += 1

        # pre-process frame
        frame = frame[..., ::-1]
        frame = image_utils.scale_image(frame)
        frame = image_utils.center_crop(frame)
        inp = image_utils.img_transform(frame)
        inp = inp[None, :, :, :]
        if use_gpu:
            inp = inp.cuda()

        # inference
        pred = model(inp)

        # post-process prediction
        pred = pred.cpu()[0].data.numpy()
        pred = image_utils.depth_to_grayscale(pred)

        # concatenate the input frame with the prediction and display
        cv2.imshow('video', np.concatenate((frame[..., ::-1], pred), axis=1))

    end = time.time()
    print('\n{} frames evaluated in {:.3f}s'.format(int(frame_cnt), end-start))
    print('{:.2f} FPS'.format(frame_cnt/(end-start)))

    capture.release()
    cv2.destroyAllWindows()


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, type=str, help='Path to the input video')
    return parser.parse_args()


def main():
    args = get_arguments()
    run_vid(args.input_path)
    

if __name__ == '__main__':
    main()
