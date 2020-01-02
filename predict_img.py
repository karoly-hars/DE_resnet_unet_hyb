import torch
import argparse
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
    print('Running the image through the network...')
    img = image_utils.load_standalone_image(img_path)

    # inference
    if use_gpu:
        img = img.cuda()

    output = model(img)
    
    # transform and plot the results
    output = output.cpu()[0].data.numpy()
    img = img.cpu()[0].data.numpy()
    image_utils.show_img_and_pred(img, output)
    

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str,  help='path to the RGB image input')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    predict_img(args.img_path)
