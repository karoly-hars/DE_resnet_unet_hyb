import os
import sys
import cv2
import tarfile
import numpy as np
import torch
import torch.nn.functional as F
from network import ResnetUnetHybrid
import image_utils


def collect_test_files(download_path='./NYU_depth_v2_test_set.tar.gz'):
    """Download the test subset, uncompress and return the image and label paths."""
    # download tar from Dropbox
    if not os.path.exists(download_path):
        print('Downloading test set...')
        os.system('wget https://www.dropbox.com/s/zq0kf40bs3gl50t/NYU_depth_v2_test_set.tar.gz')

    test_dir = './NYU_depth_v2_test_set'

    # uncompress
    if not os.path.exists(test_dir):
        print('Extracting test set...')
        tar = tarfile.open(download_path)
        tar.extractall(path='.')
        tar.close()

    # list test images and labels
    test_img_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('_rgb.png')]
    test_label_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('_dpth.npy')]
    test_img_paths.sort()
    test_label_paths.sort()

    return test_img_paths, test_label_paths


def compute_errors():
    """Download the test files, run all the test images through the model, and evaluate."""
    # switch to CUDA device if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    # load model
    print('Loading model...')
    model = ResnetUnetHybrid.load_pretrained(device=device)
    model.eval()

    preds = np.zeros((466, 582, 654), dtype=np.float32)
    labels = np.zeros((466, 582, 654), dtype=np.float32)

    test_img_paths, test_label_paths = collect_test_files()

    print('\nRunning evaluation:')
    for idx, (img_path, label_path) in enumerate(zip(test_img_paths, test_label_paths)):
        sys.stdout.write('\r{} / {}'.format(idx+1, len(test_img_paths)))
        sys.stdout.flush()

        # load image
        img = cv2.imread(img_path)[..., ::-1]

        # resize and center crop to input size
        img = image_utils.scale_image(img, 0.55)
        img = image_utils.center_crop(img)
        img = image_utils.img_transform(img)
        img = img[None, :, :, :].to(device)

        # inference
        pred = model(img)

        # up-sampling
        pred = F.interpolate(pred, size=(466, 582), mode='bilinear', align_corners=False)
        pred = pred.cpu().data.numpy()

        # load label
        label = np.load(label_path)
        # center crop to output size
        label = label[7:label.shape[0]-7, 29:label.shape[1]-29]

        # store the label and the corresponding prediction
        labels[:, :, idx] = label
        preds[:, :, idx] = pred[0, 0, :, :]

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
    compute_errors()


if __name__ == '__main__':
    main()
