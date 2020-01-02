import random
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import HEIGHT, WIDTH


def random_scale_image_end_depth(image, depth):
    r_scale = random.uniform(1.0, 1.5)
    image = scale_image(image, r_scale)
    depth = scale_image(depth, r_scale)
    depth = depth / r_scale
    return image, depth


def scale_image(image, scale=None):
    # if scale is None, scale to the longer size
    if scale is None:
        scale = max(WIDTH / image.shape[1], HEIGHT / image.shape[0])

    new_size = (math.ceil(image.shape[1] * scale), math.ceil(image.shape[0] * scale))
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    return image


def random_rotate(image, depth):
    random_rot = random.uniform(-5, 5)
    num_rows, num_cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), random_rot, 1)
    image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows), flags=cv2.INTER_NEAREST)
    depth = cv2.warpAffine(depth, rotation_matrix, (num_cols, num_rows), flags=cv2.INTER_NEAREST)
    return image, depth


def random_crop(image, depth):
    bottom = random.randint(0, (image.shape[0] - HEIGHT))
    left = random.randint(0, (image.shape[1] - WIDTH))
    corner = (bottom, left)
    image = image[corner[0]:corner[0] + HEIGHT, corner[1]:corner[1] + WIDTH]
    depth = depth[corner[0]:corner[0] + HEIGHT, corner[1]:corner[1] + WIDTH]
    return image, depth


def add_noise_to_colors(image):
    image = image * (random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2))
    # clip back to 0-255
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


def random_flip(image, depth):
    do_flip = random.random() < 0.5
    if do_flip:
        image = np.flip(image, axis=1)
        depth = np.flip(depth, axis=1)
    return image, depth


def center_crop(img):
    corner = ((img.shape[0] - HEIGHT) // 2, (img.shape[1] - WIDTH) // 2)
    img = img[corner[0]:corner[0] + HEIGHT, corner[1]:corner[1] + WIDTH]
    return img


def img_transform(img):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = data_transform(img)
    return img


def depth_transform(depth):
    depth = np.expand_dims(depth, 2)
    depth = np.transpose(depth, (2, 0, 1))
    return depth


def load_standalone_image(img_path):
    img = cv2.imread(img_path)[..., ::-1]
    img = preprocess_standalone_image(img)
    return img


def preprocess_standalone_image(img):
    img = scale_image(img)
    img = center_crop(img)
    img = img_transform(img)
    img = img[None, :, :, :]
    return img


def show_img_and_pred(img, pred):
    plt.figure()
    plt.subplot(1, 2, 1)
    img = postprocess_image(img)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    pred = np.transpose(pred, (1, 2, 0))
    plt.imshow(pred[:, :, 0])
    plt.show()


def postprocess_image(img):
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (std * img + mean)
    img = np.clip(img, 0, 1)
    return img


def depth_image_to_grayscale(depth, max_dist=10.0):
    depth = np.transpose(depth, (1, 2, 0))
    depth[depth > max_dist] = max_dist
    depth = depth / max_dist

    depth = np.array(depth * 255.0, dtype=np.uint8)
    depth = cv2.resize(depth, (WIDTH, HEIGHT))

    bgr_depth_img = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    bgr_depth_img = np.clip(bgr_depth_img, 0, 255)
    return bgr_depth_img
