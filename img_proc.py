from PIL import Image
import numpy as np
import os
import argparse
from pathlib import Path

def resize(img):
    return img.reduce(2, box=(76, 1, 524, 449)) if img.size == (600, 450) else img

def flatten_pixels(img_list):
    return [color_val for pixel in img_list for color_val in pixel]

def num_features():
    return 224 * 224 * 3

def num_examples(dir_path):
    return num_examples_class(dir_path, 'Melanoma') + num_examples_class(dir_path, 'NotMelanoma')

def num_examples_class(dir_path, classname):
    top_dir = Path(dir_path)
    class_dir_path = top_dir / classname
    class_dir = os.listdir(class_dir_path)
    return len(class_dir)

def slice_data_sequential(dir_path, batch_size):
    """
    slice_data_sequential returns a generator that allows iteration through the
    melanoma dataset at dir_path
    Args:
        dir_path: directory containing train, validate, or test data. Must have
            subdirectories named 'Melanoma' and 'NotMelanoma'
        batch_size: size of batch returned on each iteration of the generator
    Returns:
        generator that itself returns (x_data, y_data) on each iteration
    """
    top_dir = Path(dir_path)
    true_dir_path = top_dir / 'Melanoma'
    false_dir_path = top_dir / 'NotMelanoma'
    true_dir = os.listdir(true_dir_path)
    false_dir = os.listdir(false_dir_path)
    true_iter = iter(true_dir)
    false_iter = iter(false_dir)

    true_counter = 0
    false_counter = 0
    batch_counter = 0
    batch_x = []
    batch_y = []

    while (true_counter < len(true_dir) or false_counter < len(false_dir)):
        if batch_counter == batch_size:
            yield (np.array(batch_x), np.array(batch_y))
            batch_counter = 0
            batch_x = []
            batch_y = []

        if batch_counter < batch_size and true_counter < len(true_dir):
            with Image.open(true_dir_path / next(true_iter)) as img:
                batch_x.append(flatten_pixels(list(img.getdata())))
                batch_y.append(1)
            true_counter += 1
            batch_counter += 1

        if batch_counter < batch_size and false_counter < len(false_dir):
            with Image.open(false_dir_path / next(false_iter)) as img:
                batch_x.append(flatten_pixels(list(img.getdata())))
                batch_y.append(0)
            false_counter += 1
            batch_counter += 1
    
    if len(batch_x) > 0:
        yield (np.array(batch_x), np.array(batch_y))

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
