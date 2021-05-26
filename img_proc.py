from PIL import Image
import numpy as np
import os
import argparse
import random
from pathlib import Path
from keras.utils import Sequence

## TODO: change all instances of 224 to constant var

def resize(img):
    return img.reduce(2, box=(76, 1, 524, 449)) if img.size == (600, 450) else img

def flatten_pixels(img_list):
    return [color_val for pixel in img_list for color_val in pixel]

def unflatten_image(img_list):
    return [img_list[i*224:(i+1)*224] for i in range(224)]

def num_features():
    return 224 * 224 * 3

## TODO: get rid of or change
def num_examples(dir_path):
    return num_examples_class(dir_path, 'Melanoma') + num_examples_class(dir_path, 'NotMelanoma')

def num_examples_class(dir_path, classname):
    top_dir = Path(dir_path)
    class_dir_path = top_dir / classname
    class_dir = os.listdir(class_dir_path)
    return len(class_dir)

# make generator class that accepts a shuffle parameter, batch size, directory, etc.
# in init, we look at all the file names. prepend Melanoma or NotMelanoma to each file name if shuffle is true, we randomize the list. if false, we sort it
# in len, we do num indices / batch size
# have function that is get labels in order

def normalize(X):
	m = np.shape(X)[0] # number of examples
	n = np.shape(X)[1] # number of features in an example 
	mu = np.reshape(np.sum(X,axis=0),(1,n))/m
	return (X - mu)/255
	# Sigma = np.cov(X.T)
	# return np.solve(Sigma,X_centred)

class Data_Generator(Sequence):
    """
    Data_Generator
    """
    def __init__(self, dir_path, batch_size, shuffle=True, flatten=False):
        self.data_dir = Path(dir_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flatten = flatten
        self.xIDs = []
        self.labels = {}

        true_ids = ['Melanoma/' + filename for filename in os.listdir(self.data_dir / 'Melanoma')]
        false_ids = ['NotMelanoma/' + filename for filename in os.listdir(self.data_dir / 'NotMelanoma')]

        for id in true_ids:
            self.labels[id] = 1

        for id in false_ids:
            self.labels[id] = 0

        self.xIDs = true_ids + false_ids

        if shuffle:
            random.shuffle(self.xIDs)
        else:
            self.xIDs.sort()
        
    def labels(self):
        return np.array([self.labels[xID] for xID in self.xIDs])

    def __len__(self):
        return len(self.xIDs) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.xIDs)

    def __data_generation(self, xID_list):
        x_shape = (self.batch_size, 224*224*3) if self.flatten else (self.batch_size, 224, 224, 3)
        batch_x = np.empty(x_shape)
        batch_y = np.empty((self.batch_size, 1), dtype=int)

        for i, xID in enumerate(xID_list):
            with Image.open(self.data_dir / xID) as img:
                ## TODO: normalize data - do it not for each rgb if flatten but do it for each rgb if not flatten
                batch_x[i,] = np.array(flatten_pixels(list(img.getdata())) if self.flatten else unflatten_image(list(img.getdata())))
                batch_y[i,] = self.labels[xID]

        return batch_x, batch_y

    def __getitem__(self, index):
        x, y = self.__data_generation(self.xIDs[index*self.batch_size:(index+1)*self.batch_size])
        return x, y

# gets data from dir_path_src, resizes it to 224x224, and saves it in dir_path_dest
def process_data(dir_path_src, dir_path_dest):
    """
    gets data from dir_path_src, resizes it to 224x224, and saves it in dir_path_dest
    Args:
        dir_path_src: directory containing train, validate, or test data. Must have
            subdirectories named 'Melanoma' and 'NotMelanoma'
        dir_path_dest: directory to contain resized train, validate, or test data.
            Must have subdirectories named 'Melanoma' and 'NotMelanoma'
    """
    top_dir_src = Path(dir_path_src)
    true_dir_path_src = top_dir_src / 'Melanoma'
    false_dir_path_src = top_dir_src / 'NotMelanoma'
    true_iter = iter(os.listdir(true_dir_path_src))
    false_iter = iter(os.listdir(false_dir_path_src))

    top_dir_dest = Path(dir_path_dest)
    true_dir_path_dest = top_dir_dest / 'Melanoma'
    false_dir_path_dest = top_dir_dest / 'NotMelanoma'

    for src_img_name in true_iter:
        with Image.open(true_dir_path_src / src_img_name) as src_img:
            sized_img = resize(src_img)
            sized_img.save(true_dir_path_dest / src_img_name)

    for src_img_name in false_iter:
        with Image.open(false_dir_path_src / src_img_name) as src_img:
            sized_img = resize(src_img)
            sized_img.save(false_dir_path_dest / src_img_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize image data to 224x224')
    parser.add_argument('data_dir')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    process_data(data_dir / 'train_sep', data_dir / 'train_sep')
    process_data(data_dir / 'valid', data_dir / 'valid')
    process_data(data_dir / 'test', data_dir / 'test')
