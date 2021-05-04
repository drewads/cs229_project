import sys

sys.path.insert(1, '..')

from img_proc import slice_data_sequential, num_data

if __name__ == '__main__':
    print(num_data('../data/train_sep'))
    print()

    (x_data, y_data) = next(slice_data_sequential('../data/train_sep', 100))
    print('x data')
    print(x_data)
    print('y data')
    print(y_data)
    print()

    for (x_data, y_data) in slice_data_sequential('../data/train_sep', 1000):
        print(x_data.shape, y_data.shape)