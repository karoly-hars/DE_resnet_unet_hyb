import torch
from torch.autograd import Variable
from utils import load_img, show_img_pred
import argparse


    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to the trained model')
    parser.add_argument('img_path', type=str,  help='path to the RGB image input')
    args = parser.parse_args()
    
    # loading model
    print('\nLoading model...')
    import net
    model = net.hyb_net(pretrained=2, load_path=args.model_path)
    
    
    # switching to GPU if possible
    use_gpu = torch.cuda.is_available()
    print('\nusing GPU:', use_gpu)
    if use_gpu:
        model = model.cuda()
            
    # setting model to evalutation mode
    model.eval()
        
    # reading image
    img = load_img(args.img_path)
    
    # running model on the image
    if use_gpu:
        img = Variable(img.cuda())
    else:
        img = Variable(img)
    output = model(img)
    
    # transforming and ploting the results
    output = output.cpu()[0].data.numpy()
    img = img.cpu()[0].data.numpy()
    show_img_pred(img, output)
    
    

if __name__ == "__main__":
    main()

