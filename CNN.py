import evaluate
import csv
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline

def model(input_shape):
    """
    input_shape: The height, width and channels as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    """

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='CNN')

    return modeldef 


def main():
	X_train, y_train = util.load_csv('training_data.csv', add_intercept=True)
	X_test, y_test = util.load_csv('test_data.csv', add_intercept=True)

	model = CNN(np.shape(X_train))
	model.compile(optimizer = "adam", loss = "binary_cross_entropy", metrics = ["accuracy"])
	model.fit(x = X_train, y = Y_train, epochs = 100, batch_size = 10)

	y_train_pred = model.predict(X_train, Y_train)
	y_test_pred = model.predict(X_test, Y_test)
    

	evaluate.ROCandAUROC(y_train_pred,y_train,'eval_train_data.txt','ROC_train_data.jpeg','AUROC_train_data.jpeg')	
	evaluate.ROCandAUROC(y_test_pred,y_test,'eval_test_data.txt','ROC_test_data.jpeg','AUROC_test_data.jpeg')

	thresh = 0.5
	tp,fn,fp,tn = counts(y_test_pred, y_test, threshold = thresh)
	acc,prec,sens,spec,F1 = stats(tp,fn,fp,tn)
	print(f"{Threshold = {thresh}}")
	print(f"{Accuracy = {acc}}")
	print(f"{Precision = {prec}}")
	print(f"{Sensitivity = {sens}}")
	print(f"{Specificity = {spec}}")
	print(f"{F1 score = {F1}}")

if __name__ == "__main__":
    main()