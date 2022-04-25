import cv2
import numpy as np
import torch
import argparse
import time
import image_utils
from network import ResnetUnetHybrid


def run_vid(input_path, output_path):
    """Load, transform, and inference the frames of a video. Save the predictions + the input frames as a video."""
    # switch to CUDA device if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    # load model
    print('Loading model...')
    model = ResnetUnetHybrid.load_pretrained(device=device)
    model.eval()

    # start video capture
    print('Inferencing video frames...')
    start = time.time()
    capture = cv2.VideoCapture(input_path)
    frame_cnt = 0

    # init video writer
    vid_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        (image_utils.WIDTH * 2, image_utils.HEIGHT)
    )

    if not capture.isOpened():
        print('ERROR: Failed to open video.')
        return -1

    while True:
        success, frame = capture.read()
        if not success:
            print('Finished.')
            break
        frame_cnt += 1

        # pre-process frame
        frame = frame[..., ::-1]
        frame = image_utils.scale_image(frame)
        frame = image_utils.center_crop(frame)
        inp = image_utils.img_transform(frame)
        inp = inp[None, :, :, :].to(device)

        # inference
        pred = model(inp)

        # postprocess prediction
        pred = pred.cpu()[0].data.numpy()
        pred = image_utils.depth_to_grayscale(pred)

        # concatenate output and input, write frame
        output_concatednated = np.concatenate((frame[..., ::-1], pred), axis=1)
        vid_writer.write(output_concatednated)

    end = time.time()
    print('\n{} frames evaluated in {:.3f}s'.format(int(frame_cnt), end-start))
    print('{:.2f} FPS'.format(frame_cnt/(end-start)))

    capture.release()
    vid_writer.release()

def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, type=str, help='Path to the input video.')
    parser.add_argument('-o', '--output_path', required=True, type=str, help='Path to the output video.')
    return parser.parse_args()


def main():
    args = get_arguments()
    run_vid(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
