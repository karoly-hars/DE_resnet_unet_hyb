import argparse
import cv2
import torch
from network import ResnetUnetHybrid
import image_utils


def predict_img(img_path): 
    # switch to GPU if possible
    use_gpu = torch.cuda.is_available()
    print('Using GPU:', use_gpu)    
    
    # load model
    print('Loading model...')
    model = ResnetUnetHybrid.load_pretrained(use_gpu=use_gpu)
    model.eval()
        
    # load image
    img = cv2.imread(img_path)[..., ::-1]
    img = image_utils.scale_image(img)
    img = image_utils.center_crop(img)
    inp = image_utils.img_transform(img)
    inp = inp[None, :, :, :]
    if use_gpu:
        inp = inp.cuda()

    # inference
    print('Running the image through the network...')
    output = model(inp)
    
    # transform and plot the results
    output = output.cpu()[0].data.numpy()
    image_utils.show_img_and_pred(img, output)
    

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str,  help='path to the RGB image input')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    predict_img(args.img_path)
