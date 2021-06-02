import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.models import Model
from pathlib import Path
import argparse

import img_proc
import evaluate

def transfer_learning(data_gen, base_model, epochs=20):
    X_input = Input(data_gen.example_shape_tensor()) #shape: 224x224x3

    X = base_model(X_input)
    
    # start making layers here
    X = ZeroPadding2D((3, 3))(X) #shape: 13x13x2048
    X = Conv2D(filters=2048, kernel_size=(7, 7), strides = (1, 1), name = 'conv0')(X) #shape: 7x7x2048
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2,2),name='max_pool0')(X) #shape: 3x3x2048

    X = ZeroPadding2D((1, 1))(X) #shape: 5x5x2048
    X = Conv2D(filters=4096, kernel_size=(3, 3), strides = (1, 1), name = 'conv1')(X) #shape: 3x3x4096
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2,2),name='max_pool1')(X) #shape: 2x2x4096

    X = Flatten()(X) #shape: 16384
    X = Dense(units=4096, activation='relu', name='fc0')(X)
    X = Dense(units=1024, activation='relu', name='fc1')(X)
    X = Dense(units=256, activation='relu', name='fc2')(X)
    X = Dense(units=32, activation='relu', name='fc3')(X)
    X = Dense(units=1, activation='sigmoid', name='fc4')(X)

    model = Model(inputs = X_input, outputs = X, name='CNN') # Total number of trainable params = 737,537
    model.compile(optimizer = "Adam", loss = 'binary_crossentropy', metrics = ["accuracy"])
    model.fit(data_gen, epochs = epochs, use_multiprocessing=True, workers=8)

    return model

def main(data_dir,BATCH_SIZE=100):
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        input_shape=(224,224,3),
        include_top=False)
    
    print('---------- Loading Training Set ----------')
    feats = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=True, flatten=False)

    print('---------- Training Model ----------')
    model = transfer_learning(feats, base_model, epochs=1)

    print('---------- Saving Model ----------')
    model.save('saved_transfer_model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='data')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    main(data_dir)