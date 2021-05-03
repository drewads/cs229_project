import imageio
import numpy as np

# Creates list of lists where each element of the list is a row
# and each element of a row is a pixel with 3 values
def img_to_3D_list(img_path):
    return imageio.imread(img_path)

def img_to_3D_np_array(img_path):
    return np.array(img_to_3D_list(img_path))

# Row major version of 3D list. Format is
# (elem[0, 0, 0], elem[0, 0, 1], elem[0, 0, 2], elem[0, 1, 0], ...)
# where the notation here is elem[row, col, depth]
def img_to_row_major_list(img_path):
    img_3D_list = img_to_3D_list(img_path)
    row_major_first_step = [pixel for row in img_3D_list for pixel in row]
    return [color_val for pixel in row_major_first_step for color_val in pixel]

def img_to_flat_list(img_path):
    return img_to_row_major_list(img_path)
    
def img_to_row_major_np_array(img_path):
    return np.array(img_to_row_major_list(img_path))

def img_to_flat_np_array(img_path):
    return img_to_row_major_np_array(img_path)