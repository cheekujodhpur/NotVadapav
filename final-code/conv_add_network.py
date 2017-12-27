import keras
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras import backend as K

import sys
import numpy as np
import pickle as pkl
import time

from get_train_test_data import getTrainTestData

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
                                allow_soft_placement=True, device_count = {'CPU': 1})
session = tf.Session(config=config)
K.set_session(session)

start_prog_time = time.time()

# Wrapper function for deprecated sigmoid_cross_entropy_with_logits usage
def wrap(f):
    def wrapper(*arg, **kw):
        kw = {'logits': arg[0], 'labels':arg[1]}
        arg=()
        ret = f(*arg, **kw)
        return ret
    return wrapper

tf.nn.sigmoid_cross_entropy_with_logits = wrap(tf.nn.sigmoid_cross_entropy_with_logits)

# Set number of vadapavs and burgers (from sys args)
NUM_VADAPAVS = int(sys.argv[1])
NUM_BURGERS = int(sys.argv[2])

print 'Number of vadapavs:', NUM_VADAPAVS
print 'Number of burgers:', NUM_BURGERS

train_set, test_set = getTrainTestData(NUM_VADAPAVS, NUM_BURGERS)

TRAINING_SIZE = len(train_set)
TESTING_SIZE = len(test_set)

#Randomly permutes order of data
np.random.shuffle(train_set)
np.random.shuffle(test_set)

# print 'Getting training and testing sets...'
train_X = [x[0] for x in train_set]
train_Y = [x[1] for x in train_set]
train_X = np.asarray(train_X)
train_Y = np.asarray(train_Y)

test_X = [x[0] for x in test_set]
test_Y = [x[1] for x in test_set]
test_X = np.asarray(test_X)
test_Y = np.asarray(test_Y)

#Using inbuilt VGG19 model with weights pretrained on imaegnet data, leaving out the final 3 Fully Connected layers
pt_model = VGG19(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
model = Model(input=pt_model.input, output=pt_model.get_layer('block4_pool').output)

# Number of features extracted from block4_pool
EXTRACTED_FEATURE_SHAPE = (32,32,512)

def getB4PFeatures(data):

    data_b4pool = []
    for i in range(len(data)):

        input_data = data[i]
        b4p_features = model.predict(input_data)
        b4p_features = b4p_features.reshape(EXTRACTED_FEATURE_SHAPE)
        data_b4pool.append(b4p_features)

    return np.asarray(data_b4pool)

# print 'Getting B4P features for train and test...'
train_X = getB4PFeatures(train_X)
test_X = getB4PFeatures(test_X)

vgg_wts = pt_model.get_weights()

# Defining our new Sequential model (to be trained)
new_model = Sequential()
new_model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu', input_shape =(32,32,512), name='block5_conv1', weights = [vgg_wts[24],vgg_wts[25]]))
new_model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='block5_conv2',weights = [vgg_wts[26],vgg_wts[27]]))
new_model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='block5_conv3',weights = [vgg_wts[28],vgg_wts[29]]))
new_model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='block5_conv4',weights = [vgg_wts[30],vgg_wts[31]]))
new_model.add(MaxPooling2D((2,2), strides=(2,2), name='block5_pool'))

new_model.add(Flatten()) 
new_model.add(Dense(32, activation='sigmoid' ))
new_model.add(Dense(32, activation='sigmoid'))
new_model.add(Dense(1, activation='sigmoid'))
new_model.compile(optimizer=SGD(lr=0.01, decay=1e-5, momentum=0, nesterov=False),loss='binary_crossentropy', metrics=['accuracy'])

# Train the new model
training_start = time.time()
new_model.fit(train_X, train_Y, nb_epoch=20, batch_size=32)

training_end = time.time()
print 'Training time:', (training_end-training_start)/60

new_model.save('v'+str(NUM_VADAPAVS)+'-b'+str(NUM_BURGERS)+'-conv_bestmodel.h5')

# Get the confusion matrix for the test dataset
conf_matrix = {}
for i in range(2):
    for j in range(3):
        conf_matrix[(i,j)] = 0

predictions = new_model.predict(test_X)
for i in range(TESTING_SIZE):
    
    # test_inp = test_X[i].reshape(B4P_FEATURE
    target = test_Y[i]
    # logit = new_model.predict(test_inp)
    pred = predictions[i]
    if pred >= 0.5:
        output = 1
    else:
        output = 0
    conf_matrix[(output, target)] += 1

correct_pred = conf_matrix[(0,0)]+conf_matrix[(1,1)]+conf_matrix[(0,2)]
score = (1.0*correct_pred)/TESTING_SIZE
print 'Accuracy:', score

print 'Confusion matrix:'
print '\t0\t1\t2'
for i in range(2):
    print i,'\t',
    for j in range(3):
       print conf_matrix[(i,j)],'\t',
    print

print '-------------------------------------------------'

# Export results
#with open('conv_results.txt', 'a') as results_fp:
#
#    results_fp.write('----------------------\n')
#    results_fp.write('Vadapavs:'+str(NUM_VADAPAVS)+'\tBurgers:'+str(NUM_BURGERS)+'\n')
#    results_fp.write('Accuracy:'+str(score*100)+'%\n')
#    results_fp.write('Confusion_matrix:\n')
#    results_fp.write('\t0\t1\t2\n')
#    results_fp.write('0\t'+str(conf_matrix[(0,0)])+'\t'+str(conf_matrix[(0,1)])+'\t'+str(conf_matrix[(0,2)])+'\n')
#    results_fp.write('1\t'+str(conf_matrix[(1,0)])+'\t'+str(conf_matrix[(1,1)])+'\t'+str(conf_matrix[(1,2)])+'\n')
#    results_fp.write('----------------------\n')

