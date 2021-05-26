import evaluate
import img_proc
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
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model
# from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# %matplotlib inline

def CNN(X_train,y_train,batch_size = 20,epochs = 10):
    """
   
    """

    X_input = Input(X_train.shape[1:]) #shape: 224x224x3

    X = ZeroPadding2D((3, 3))(X_input) #shape: 230x230x3
    X = Conv2D(filters=32, kernel_size=(7, 7), strides = (1, 1), name = 'conv0')(X) #shape: 224x224x32
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='max_pool0')(X) #shape: 112x112x32

    X = ZeroPadding2D((2, 2))(X_input) #shape: 116x116x32
    X = Conv2D(filters=64, kernel_size=(5, 5), strides = (1, 1), name = 'conv1')(X) #shape: 112x112x64
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='max_pool1')(X) #shape: 56x56x64

    X = ZeroPadding2D((1, 1))(X_input) #shape: 58x58x64
    X = Conv2D(filters=128, kernel_size=(3, 3), strides = (1, 1), name = 'conv2')(X) #shape: 56x56x128
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='max_pool2')(X) #shape: 28x28x128

    X = Conv2D(filters=128, kernel_size=(3, 3), strides = (3, 3), name = 'conv3')(X) #shape: 9x9x128
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3,3), strides=(3,3),name='max_pool3')(X) #shape: 3x3x128


    X = Flatten()(X) #shape: 1152
    X = Dense(units=512, activation='relu', name='fc0')(X)
    X = Dense(units=32, activation='relu', name='fc1')(X)
    X = Dense(units=1, activation='sigmoid', name='fc2')(X)

    model = Model(inputs = X_input, outputs = X, name='CNN') # Total number of trainable params = 737,537
    model.compile(optimizer = "Adam", loss = 'binary_crossentropy', metrics = ["accuracy"])
    model.fit(x = X_train, y = y_train, epochs = epochs, batch_size = batch_size)

    return model 

def normalize(X):
    m = np.shape(X)[0] # number of examples
    n_H = np.shape(X)[1] # number of features in an example 
    n_W = np.shape(X) [2]
    n_C = np.shape(X)[3]
    mu = np.reshape(np.sum(X,axis=0),(n_H,n_W,n_C))/m
    print(f"Shape normalized X = {((X - mu)/255).shape}")
    return (X - mu)/255

def main():
    data_train = img_proc.Data_Generator('data/train_sep', 1000, shuffle=True, flatten=False)
    X_train, y_train = data_train.__getitem__(1)
    print(f"X_train.shape = {X_train.shape}")
    print(f"y_train.shape = {y_train.shape}")
    model = CNN(normalize(X_train),y_train, batch_size = 25, epochs = 5)
    y_train_pred = model.predict(X_train)

    data_valid = img_proc.Data_Generator('data/valid', 1000, shuffle=True, flatten=False)
    X_valid, y_valid = data_valid.__getitem__(1)
    y_valid_pred = model.predict(normalize(X_valid))
    

    auc_roc,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_cnn.jpeg')

    print(f"\nArea Under ROC = {auc_roc}")
    tp,fn,fp,tn = evaluate.counts(y_train_pred, y_train, threshold = threshold_best)
    acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    print("\nStats for predictions on train set:")
    print(f"Threshold = {threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")

    tp,fn,fp,tn = evaluate.counts(y_valid_pred, y_valid, threshold = threshold_best)
    acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    print("\nStats for predictions on validation set:")
    print(f"Threshold = {threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")

if __name__ == "__main__":
    main()