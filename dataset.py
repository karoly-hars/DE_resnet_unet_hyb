import os
import cv2
import torch
from torch.utils.data import Dataset
import image_utils

HEIGHT = 256
WIDTH = 320

OUT_WIDTH = 160
OUT_HEIGHT = 128


class RGBDDataset(Dataset):

    def __init__(self, data_dir, subset):
        self.subset = subset
        assert subset in ['train', 'val']

        self.img_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_rgb.png')]
        self.dpth_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_dpth.png')]
        self.img_paths.sort()
        self.dpth_paths.sort()
        assert len(self.img_paths) == len(self.dpth_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx][0])[..., ::-1]
        depth = cv2.imread(self.dpth_paths[idx][1], -1)

        image, depth = self.augment_image(image, depth)

        image = image_utils.img_transform(image)
        depth = image_utils.depth_transform(depth)

        image = torch.tensor(image)
        depth = torch.tensor(depth)

        yield image, depth

    def augment_image(self, image, depth):
        # resize to 320x240
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (depth.shape[1] // 2, depth.shape[0] // 2), interpolation=cv2.INTER_NEAREST)

        if self.subset == 'train':
            image, depth = image_utils.random_scale_image_end_depth(image, depth)
            image, depth = image_utils.random_rotate(image, depth)
            image, depth = image_utils.random_crop(image, depth)
            image, depth = image_utils.add_noise_to_colors(image)
            image, depth = image_utils.random_flip(image, depth)
        else:
            image = image_utils.center_crop(image)
            depth = image_utils.center_crop(depth)

        # extra resize just in case and return
        if image.shape != (WIDTH, HEIGHT, 3):
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        if depth.shape != (OUT_WIDTH, OUT_HEIGHT, 1):
            depth = cv2.resize(depth, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_NEAREST)

        return image, depth
