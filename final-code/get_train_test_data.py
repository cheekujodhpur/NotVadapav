import os
import time
import sys

import numpy as np
import pickle

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input


def preprocDataset(raw_image_list, DIRECTORY, tag):

    '''
    Return list of 2-tuples: (img_array, tag)
    img_array: numpy array, one for each image; images resized to 512x512
    tag: image class (0-not vadapav, 1-vadapav, 2-burger(0 for training)
    '''

    list_images = []
    for img_file in raw_image_list:

        filename = DIRECTORY+img_file
        img = image.load_img(filename, target_size=(512, 512))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        list_images.append((x, tag))

    return list_images


def getTrainTestData(NUM_VADAPAVS, NUM_BURGERS):

    '''
    For each of vadapav, not-vadapav and burger classes
    last 20% of all available images are added to a test set
    among first 80%, first NUM_VADAPAV/all/NUM_BURGERS images are taken

    train_set, test_set: list of 2-tuples (img_array, tag) to be used for training and testing
    '''

    VADAPAV_DIR = 'data/v/'
    NOT_VADAPAV_DIR = 'data/nv/'
    BURGER_DIR = 'data/b/'

    # Get list of image filenames for each class
    list_v_images = os.listdir(VADAPAV_DIR)
    list_nv_images = os.listdir(NOT_VADAPAV_DIR)
    list_b_images = os.listdir(BURGER_DIR)

    # Get train and test images for vadapavs
    num_total_v = len(list_v_images)
    v_test_images = list_v_images[int(0.8*num_total_v):]
    rem_v_images = list_v_images[:int(0.8*num_total_v)]
    np.random.shuffle(rem_v_images)
    v_train_images = rem_v_images[:NUM_VADAPAVS]

    # Get train and test images for not vadapavs
    num_total_nv = len(list_nv_images)
    nv_test_images = list_nv_images[int(0.8*num_total_nv):]
    rem_nv_images = list_nv_images[:int(0.8*num_total_nv)]
    np.random.shuffle(rem_nv_images)
    nv_train_images = rem_nv_images

    # Get train and test images for burgers
    num_total_b = len(list_b_images)
    b_test_images = list_b_images[int(0.8*num_total_b):]
    rem_b_images = list_b_images[:int(0.8*num_total_b)]
    np.random.shuffle(rem_b_images)
    b_train_images = rem_b_images[:NUM_BURGERS]

    list_images = []

    start_time = time.time()
    
    # Preprocess the vadapav images
    v_train_proc_list = preprocDataset(v_train_images, VADAPAV_DIR, 1)
    v_test_proc_list = preprocDataset(v_test_images, VADAPAV_DIR, 1)
    print 'Finished vadapavs...'

    # Preprocess the not vadapav images
    nv_train_proc_list = preprocDataset(nv_train_images, NOT_VADAPAV_DIR, 0)
    nv_test_proc_list = preprocDataset(nv_test_images, NOT_VADAPAV_DIR, 0)
    print 'Finished not vadapavs...'

    # Preprocess the burger images
    b_train_proc_list = preprocDataset(b_train_images, BURGER_DIR, 0)
    b_test_proc_list = preprocDataset(b_test_images, BURGER_DIR, 2)
    print 'Finished burgers...'

    # Combine class train and test lists to get a single train and test set
    train_list = v_train_proc_list + nv_train_proc_list + b_train_proc_list
    test_list = v_test_proc_list + nv_test_proc_list + b_test_proc_list

    print 'Training size:', len(train_list)
    print 'Testing size:', len(test_list)

    return train_list, test_list


