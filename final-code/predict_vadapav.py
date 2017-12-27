import keras
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense
from keras import backend as K

import sys
import numpy as np
import pickle
import time

from get_train_test_data import preprocDataset

# To use the GPU
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
                                allow_soft_placement=True, device_count = {'CPU': 1})
session = tf.Session(config=config)
K.set_session(session)


# Wrapper function for deprecated sigmoid_cross_entropy_with_logits usage
def wrap(f):
    def wrapper(*arg, **kw):
        kw = {'logits': arg[0], 'labels':arg[1]}
        arg=()
        ret = f(*arg, **kw)
        return ret
    return wrapper

tf.nn.sigmoid_cross_entropy_with_logits = wrap(tf.nn.sigmoid_cross_entropy_with_logits)

# Get filename of image to classify
filename = sys.argv[1]

# Preprocess the image
preprocd_image = preprocDataset([filename], '', 0)[0][0]

# Number of features extracted from block4_pool layer
EXTRACTED_FEATURE_SHAPE = (1, 32, 32, 512) 

# Define pre-trained VGG19 model
pt_model = VGG19(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
model = Model(input=pt_model.input, output=pt_model.get_layer('block4_pool').output)

# Extract block5_pool features for the input image
b5p_features = model.predict(preprocd_image)
b5p_features = b5p_features.reshape(EXTRACTED_FEATURE_SHAPE)

# Load our saved model (best conv model)
saved_model = load_model('v650-b120-conv-bestmodel.h5')

# Get prediction from saved model
pred = saved_model.predict(b5p_features)

# Classify
if pred >= 0.5:
    output = 1
else:
    output = 0

print 'Output:', output

