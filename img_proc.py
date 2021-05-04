from PIL import Image
import numpy as np
import os
from pathlib import Path

def resize(img):
    return img.reduce(2, box=(76, 1, 524, 449)) if img.size == (600, 450) else img

def flatten_pixels(img_list):
    return [color_val for pixel in img_list for color_val in pixel]

def num_data(dir_path):
    top_dir = Path(dir_path)
    true_dir_path = top_dir / 'Melanoma'
    false_dir_path = top_dir / 'NotMelanoma'
    true_dir = os.listdir(true_dir_path)
    false_dir = os.listdir(false_dir_path)
    return len(true_dir) + len(false_dir)

def slice_data_sequential(dir_path, batch_size):
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
                sized_img = resize(img)
                batch_x.append(flatten_pixels(list(sized_img.getdata())))
                batch_y.append(1)
            true_counter += 1
            batch_counter += 1

        if batch_counter < batch_size and false_counter < len(false_dir):
            with Image.open(false_dir_path / next(false_iter)) as img:
                sized_img = resize(img)
                batch_x.append(flatten_pixels(list(sized_img.getdata())))
                batch_y.append(0)
            false_counter += 1
            batch_counter += 1
    
    if len(batch_x) > 0:
        yield (np.array(batch_x), np.array(batch_y))