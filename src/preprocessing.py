import os
import numpy as np
from six.moves import urllib
from scipy import ndimage, misc
import matplotlib.image as mpimg
from skimage.transform import resize
import sys
import argparse

def get_args():
    """
    Parses arguments passed in command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        default = './models/'
    )
    parser.add_argument(
        '--job-name',
        type=str,
        default=''
    )
    parser.add_argument(
        '--load_best',
        dest = 'load_best',
        action= 'store_true'
    )
    parser.add_argument(
        '--train',
        dest = 'load_best',
        action= 'store_false'
    )
    parser.set_defaults(load_best=True)
    args, _ = parser.parse_known_args()
    return args


def load_all_imgs(num_images, padding_size, data_path):
    x_imgs = []
    y_imgs = []

    # Load all images
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        x_image_filename = data_path + '/training/images/' + imageid + '.png'
        y_image_filename = data_path + '/training/groundtruth/' + imageid + '.png'

        if os.path.isfile(x_image_filename) and os.path.isfile(y_image_filename):
            x_img = mpimg.imread(x_image_filename)
            x_img = add_padding(x_img, padding_size, 3)
            x_imgs.append(x_img)

            y_img = mpimg.imread(y_image_filename)
            y_img = y_img.reshape((y_img.shape[0], y_img.shape[1], 1))
            y_img = add_padding(y_img, padding_size, 1)
            y_imgs.append(y_img)

        else:
            print('File ' + x_image_filename + ' does not exist') 
    
    return x_imgs, y_imgs


def data_generator(patch_size, num_images = 100, train_test_ratio = 0.8, padding_size = 28,
                   data_path = "./data"):
    """
    Load all of images and return the x_train, x_test, y_train and y_test
    """
        
    x_imgs, y_imgs = load_all_imgs(num_images, padding_size, data_path)
    
    num_images = len(x_imgs)
    x_train, x_test, y_train, y_test = patches_split(x_imgs, y_imgs, patch_size, train_test_ratio, padding_size)

    y_train = prepare_labels(y_train)
    y_test = prepare_labels(y_test)

    return x_train, x_test, y_train, y_test


def add_padding(img, padding_size, channels):
    """
    Adds padding to an image and return one padded image
    """
    padded_img = np.zeros((img.shape[0] + padding_size*2,
                           img.shape[1] + padding_size*2,
                           channels))
    for channel in range(channels):
        padded_img[:,:,channel] = np.pad(img[:,:,channel],
                                         ((padding_size, padding_size),(padding_size, padding_size)), 
                                         'symmetric')
    return padded_img

def img_crop(im, w, h, p):
    """
    Crops image with respect to width/height of patches and padding and return the array of patches
    """
    list_patches = []
    imgwidth = im.shape[0] - p*2
    imgheight = im.shape[1] - p*2
    is_2d = len(im.shape) < 3
    for i in range(p,imgheight+p,h):
        for j in range(p,imgwidth+p,w):
            if is_2d:
                im_patch = im[j-p:j+w+p, i-p:i+h+p]
            else:
                im_patch = im[j-p:j+w+p, i-p:i+h+p, :]
            list_patches.append(im_patch)
    return list_patches


def patches_split(x, y, patch_size, split, padding_size):
    """
    Splits x and y into train/test patches and return x_train, x_test, y_train, y_test
    """
    
    perm = np.random.permutation(len(x))
    split_perm = int(len(x)*split)
    train_perm = perm[:split_perm]
    test_perm = perm[split_perm:]
    
    x_train_patches = [img_crop(x[i], patch_size, patch_size, padding_size) for i in train_perm]
    x_train = np.asarray([x_train_patches[i][j] for i in range(len(x_train_patches)) for j in range(len(x_train_patches[i]))])
    x_test_patches = [img_crop(x[i], patch_size, patch_size, padding_size) for i in test_perm]
    x_test = np.asarray([x_test_patches[i][j] for i in range(len(x_test_patches)) for j in range(len(x_test_patches[i]))])
    
    y_train_patches = [img_crop(y[i], patch_size, patch_size, padding_size) for i in train_perm]
    y_train = np.asarray([y_train_patches[i][j] for i in range(len(y_train_patches)) for j in range(len(y_train_patches[i]))])
    y_test_patches = [img_crop(y[i], patch_size, patch_size, padding_size) for i in test_perm]
    y_test = np.asarray([y_test_patches[i][j] for i in range(len(y_test_patches)) for j in range(len(y_test_patches[i]))])
    
    return x_train, x_test, y_train, y_test


def prepare_labels(y):
    """
    road = 1
    others = 0
    """
    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    return y.astype(int)


def patches_to_images(patches, patch_size, img_side_len=400):
    """
    Transforms patches into full images and return the array of images
    """
    
    num_patches_img = (img_side_len/patch_size) ** 2
    num_imgs = int(patches.shape[0]/num_patches_img)
    imgs = []
    
    tot_index = 0
    for img in range(num_imgs):
        img_index = 0
        image = []
        for row in range(int(np.sqrt(num_patches_img))):
            img_row = []
            for col in range(int(np.sqrt(num_patches_img))):
                if len(img_row)==0:
                    img_row = patches[tot_index]
                else:
                    img_row = np.append(img_row, patches[tot_index], axis=0)
                tot_index += 1
            if len(image)==0:
                image = img_row
            else:
                image = np.append(image, img_row, axis=1)
        imgs.append(image)
        
    return np.asarray(imgs)
