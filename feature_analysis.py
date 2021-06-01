import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

def grayscale(vector):
    min_val = min(vector)
    max_val = max(vector)
    grayscale_vector = np.empty(len(vector))
    for i in range(len(vector)):
        grayscale_vector[i] = (vector[i]-min_val)/(max_val-min_val)
    return grayscale_vector.reshape((224,224))

def main():
    model = keras.models.load_model('savedCNN_data')
    for layer in model.layers:
        weights = layer.get_weights()

    

    weights = np.genfromtxt('log_reg_weights.csv', delimiter=",")
    red_weights = [weights[i] for i in range(0,len(weights),3)]
    green_weights = [weights[i] for i in range(1,len(weights),3)]
    blue_weights = [weights[i] for i in range(2,len(weights),3)]
    np.savetxt("red_log_reg_weights.csv", red_weights, delimiter=",")
    np.savetxt("green_log_reg_weights.csv", green_weights, delimiter=",")
    np.savetxt("blue_log_reg_weights.csv", blue_weights, delimiter=",")

    grayscale_red = grayscale(red_weights)
    grayscale_green = grayscale(green_weights)
    grayscale_blue = grayscale(blue_weights)
    # print(grayscale_red[0])
    # tmp = np.empty(224)
    # for i in range(224):
    #     tmp[i] = (red_weights[i]-min(red_weights))/(max(red_weights)-min(red_weights))
    # print(tmp[0:224])

    # Creates PIL image
    img = Image.fromarray(np.uint8(grayscale_red * 255), 'L')
    img.show()

    # Creates PIL image
    img = Image.fromarray(np.uint8(grayscale_green * 255), 'L')
    img.show()

    # Creates PIL image
    img = Image.fromarray(np.uint8(grayscale_blue * 255), 'L')
    img.show()



if __name__ == '__main__':
    main()