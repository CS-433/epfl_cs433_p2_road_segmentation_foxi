import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize
import re
from src.preprocessing import add_padding, img_crop, patches_to_images

foreground_threshold = 0.25
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def csv_generation(submission_name, model, padding_size = 0, patch_size = 400):
    """
    Create the submission '.csv' file.
    """
    
    # Load images
    test_set = load_test_img(padding_size = padding_size)
    
    # Get patches 
    img_patches = [img_crop(test_set[i], patch_size, patch_size, padding_size) for i in range(len(test_set))]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    
    # Predict on given model
    pred_patches = model.predict(img_patches)
    # Reassemble to img
    predictions = patches_to_images(np.asarray(pred_patches), patch_size, img_side_len = 600)

    
    # Fix labels
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    
    # resize the images into 608*608 resolution
    new_predictions = []
    for i in range(50):
       new_prediction = change_res(predictions[i], 1, 608)
       new_predictions.append(new_prediction)

      

    # Save predictions as imgs, and keep the names
    image_names = save_test_img(new_predictions)
    
    masks_to_submission(submission_name, image_names)
    
    print('Created a submission.csv file')

def load_test_img(filepath = './data/test_set_images/', img_size = 600, padding_size = 14):
    """
    Loads all test images on and returns them as
    """
    test_imgs =[]
    
    # Load all images
    num_images = 50
    for i in range(1, num_images+1):
        test_id = 'test_' + str(i)
        image_path = filepath + test_id + '/' + test_id + '.png'

        if os.path.isfile(image_path):
            test_img = change_res(mpimg.imread(image_path), 3, img_size)
            test_img = add_padding(test_img, padding_size, 3)
            test_imgs.append(test_img)
        else:
            print('File ' + image_path + ' does not exist')
    return np.asarray(test_imgs)


def save_test_img(pred, filepath = './prediction_output/'):
    """ 
    Saves predicted test images to the folder named prediction_output
    """
    
    image_names = []
  
    # Load all images
    num_images = 50
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
        
    for i in range(1, num_images+1):
        test_id = 'test_' + str(i)
        
        if not os.path.isdir(filepath + test_id):
            os.mkdir(filepath + test_id)
            
        image_path = filepath + test_id + '/' + test_id + '.png'
        image_names.append(image_path)
        cv2.imwrite(image_path, pred[i-1], [cv2.IMWRITE_PNG_BILEVEL, 1])
        
    return image_names

def change_res(x, channels, res):
    """
    Resize the photo
    """

    return np.asarray(resize(x, (res, res, channels)))


