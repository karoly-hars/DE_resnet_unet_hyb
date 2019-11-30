import torch
from utils import load_img, show_img_pred
import argparse
import net


def predict_img(img_path): 
    # switching to GPU if possible
    use_gpu = torch.cuda.is_available()
    print("\nusing GPU:", use_gpu)    
    
    # loading model
    print("\nLoading model...")
    model = net.load_model(use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
            
    # setting model to evaluation mode
    model.eval()
        
    # reading image
    print("\nLoading and running image...")
    img = load_img(img_path)
    
    # running model on the image
    if use_gpu:
        img = img.cuda()

    output = model(img)
    
    # transforming and plotting the results
    output = output.cpu()[0].data.numpy()
    img = img.cpu()[0].data.numpy()
    show_img_pred(img, output)
    

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str,  help="path to the RGB image input")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    predict_img(args.img_path)
