README
------------------------------------------------------

Our code runs on Keras v1.2.0 and TensorFlow v1.2.1 (versions available on the server we were using, cannot be upgraded)

Data is supposed to be in the folder 'data' in this directory, which has subdirectories 'v', 'nv', 'b', but data is not included here since size is 15 GB.

------------------------------------------------------
1) get_train_test_data.py
Used to get train and test datasets
This file is NOT to be called directly

Modules:
getTrainTestData - Returns train and test lists, for parameters NUM_VADAPAVS and NUM_BURGERS
preprocData - Image preprocessing and tagging

------------------------------------------------------

2) fc_network.py
Vadapav classifier where only fully connected layers are trained

Usage: python fc_network.py <NUM_VADAPAVS> <NUM_BURGERS>

where 
NUM_VADAPAVS is from the set {500, 550, 600, 650}
NUM_BURGERS is from the set {30, 60, 90, 120}

------------------------------------------------------

3) conv_add_network.py
Vadapav classifier where one convolutional block and three fully connected layers are trained

Usage: python conv_add_network.py <NUM_VADAPAVS> <NUM_BURGERS>

where 
NUM_VADAPAVS is from the set {500, 550, 600, 650}
NUM_BURGERS is from the set {30, 60, 90, 120}

------------------------------------------------------

4) predict_vadapav.py
For a given image file, predict if it contains a vadapav or not. Requires h5 file of our best model which could not be uploaded due to large size (100 MB).

Usage: python predict_vadapav.py <IMG_FILENAME>

where IMG_FILENAME is the name of the image file to classify. Expected to be relative to current directory.

------------------------------------------------------

5) get_fc_model_results.sh
Trains models for different values of NUM_VADAPAVS and NUM_BURGERS, by calling fc_network.py each time.

Usage: bash get_fc_model_results.sh

------------------------------------------------------

6) get_conv_model_results.sh
Trains models for different values of NUM_VADAPAVS and NUM_BURGERS, by calling conv_network.py each time.

Usage: bash get_conv_model_results.sh





