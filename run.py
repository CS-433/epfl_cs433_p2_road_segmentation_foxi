import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

from src.model_unet import UNET
from src.preprocessing import data_generator, prepare_labels, get_args
from src.csv_generation import csv_generation

# Hyperparameters
EPOCHS = 200
BATCH_SIZE = 64
DEPTH = 4
PATCH_SIZE = 200
PADDING = 28
NUM_IMAGES = 100
TRAIN_TEST_RATIO = 0.8


device = len(tf.config.experimental.list_physical_devices('GPU'))

# Load training data and validation data
x_tra, x_val, y_tra, y_val = data_generator(PATCH_SIZE, train_test_ratio = TRAIN_TEST_RATIO, 
                        num_images = NUM_IMAGES, 
                        padding_size=PADDING)

print('Loading the training dataset and validatin dataset')

# Initialize Keras Imagegenerator
datagen = ImageDataGenerator()
datagen.fit(x_tra)
get_args = get_args()

# Builds the UNET
UNET = UNET(get_args, image_shape = x_tra[0].shape, depth = DEPTH)
UNET.construct_unet(device)
UNET.trainer(datagen, x_tra, y_tra, x_val, y_val, epochs = EPOCHS, batch_size = BATCH_SIZE)

# Generate the 'submission.csv' file
csv_generation('submission.csv', UNET.model, patch_size = PATCH_SIZE, padding_size= PADDING)


