import argparse
import cv2
import torch
from network import ResnetUnetHybrid
import image_utils


def predict_img(img_path):
    """Inference a single image."""
    # switch to CUDA device if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))
    
    # load model
    print('Loading model...')
    model = ResnetUnetHybrid.load_pretrained(device=device)
    model.eval()
        
    # load image
    img = cv2.imread(img_path)[..., ::-1]
    img = image_utils.scale_image(img)
    img = image_utils.center_crop(img)
    inp = image_utils.img_transform(img)
    inp = inp[None, :, :, :].to(device)

    # inference
    print('Running the image through the network...')
    output = model(inp)
    
    # transform and plot the results
    output = output.cpu()[0].data.numpy()
    image_utils.show_img_and_pred(img, output)
    

def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', required=True, type=str, help='Path to the input image.')
    return parser.parse_args()


def main():
    args = get_arguments()
    predict_img(args.img_path)


if __name__ == '__main__':
    main()
