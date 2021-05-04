import evaluate
import csv
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def load_csv(csv_path, label_col='y', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    # Load headers
    with open(csv_path, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels

def DLNN(X,Y,nn_dims,epochs = 100,batch_size = 25,loss='binary_crossentropy'):
	""" 
	Inputs:
		 X - training data (np.array(rows: #features; cols: #examples))
		 Y - training data labels (np.array(rows: #examples; cols: 1))
		 nn_dims - list with number of neurons in each layer [#input layer, # 1st hidden layer, ..., 1 neuron with sigmoid]
		 epochs - number of iterations to run gradient descent 
		 batch_size - number of training examples to include in a batch
		 loss - loss function for gradient descent to optimize over (input as string)

	Outputs:
		trained_model - model trained in keras

	"""
	n = np.shape(X)[0] # number of features in an example
	m = np.shape(X)[1] # number of examples
	layers = len(nn_dims)[0] # number of layers

	trained_model = Sequential()
	trained_model.add(Dense(nn_dims[0], input_dim = n, activation = 'relu'))


	for i in range(1,layers - 1):
		neuons = nn_dims[i]
		trained_model.add(Dense(neurons, activation='relu'))

	trained_model.add(Dense(1, activation='sigmoid'))
	trained_model.compile(loss = loss, optimizer = 'adam', metrics=['accuracy'])
	trained_model.fit(X, Y, epochs = epochs, batch_size = batch_size)

return trained_model

def main():
	X_train, y_train = util.load_csv('training_data.csv', add_intercept=True)
	X_test, y_test = util.load_csv('test_data.csv', add_intercept=True)

	model = DLNN(X_train,y_test,[np.shape(X)[0],64,16,4,1])
	y_train_pred = model.predict(X_test)
	y_test_pred = model.predict(X_test)
    

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

