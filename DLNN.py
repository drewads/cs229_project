import evaluate
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import regularizers
# from keras.utils.vis_utils import plot_model
import img_proc

def DLNN(data,m,n,nn_dims,epochs = 500,batch_size = 25,loss='binary_crossentropy',lambd = 1e-4):
	""" 
	Inputs:
		 data - training data (np.array(rows: #examples; cols: #features), np.array(np.array(rows: #examples; cols: 1)))
		 m - number of examples
		 n - number of features
		 nn_dims - list with number of neurons in each layer [#input layer, # 1st hidden layer, ..., 1 neuron with sigmoid]
		 epochs - number of iterations to run gradient descent 
		 batch_size - number of training examples to include in a batch
		 loss - loss function for gradient descent to optimize over (input as string)

	Outputs:
		trained_model - model trained in keras

	"""
	print([m,n])
	layers = len(nn_dims) # number of layers

	trained_model = Sequential()
	trained_model.add(Dense(nn_dims[0], input_dim = n, activation = 'relu',kernel_initializer = 'glorot_uniform',bias_initializer='zeros',kernel_regularizer=regularizers.l2(lambd), bias_regularizer=regularizers.l2(lambd)))


	for i in range(1,layers - 1):
		neurons = nn_dims[i]
		trained_model.add(Dense(neurons, activation='relu',kernel_initializer = 'glorot_uniform',bias_initializer='zeros', kernel_regularizer=regularizers.l2(lambd), bias_regularizer=regularizers.l2(lambd)))

	trained_model.add(Dense(1, activation='sigmoid',kernel_initializer = 'glorot_uniform',bias_initializer='zeros',kernel_regularizer=regularizers.l2(lambd), bias_regularizer=regularizers.l2(lambd)))
	print(trained_model.summary())
	# plot_model(trained_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	trained_model.compile(loss = loss, optimizer = 'adam', metrics=['accuracy'])
	trained_model.fit(data, epochs = epochs, batch_size = batch_size)

	return trained_model

def normalize(X):
	m = np.shape(X)[0] # number of examples
	n = np.shape(X)[1] # number of features in an example 
	mu = np.reshape(np.sum(X,axis=0),(1,n))/m
	return (X - mu)/255
	# Sigma = np.cov(X.T)
	# return np.solve(Sigma,X_centred)

def main():
	# todo: allow normalization function to be passed to slice_data_sequential
	train_data = img_proc.slice_data_sequential('data_proc/train_sep', 1000)
	model = DLNN(train_data,img_proc.num_examples('data_proc/train_sep'),img_proc.num_features(),[1024,256,64,16,4,1],epochs = 20)
	y_train_pred = model.predict(train_data)

	# todo: normalization function
	valid_data = img_proc.slice_data_sequential('data/valid', 1000)
	y_valid_pred = model.predict(valid_data)
    

	# TODO: auc_roc,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_dlnn.jpeg')

	# TODO:
	# print(f"\nArea Under ROC = {auc_roc}")
	# tp,fn,fp,tn = evaluate.counts(y_train_pred, y_train, threshold = threshold_best)
	# acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
	# print("\nStats for predictions on train set:")
	# print(f"Threshold = {threshold_best}")
	# print(f"Accuracy = {acc}")
	# print(f"Precision = {prec}")
	# print(f"Sensitivity = {sens}")
	# print(f"Specificity = {spec}")
	# print(f"F1 score = {F1}")

	# tp,fn,fp,tn = evaluate.counts(y_valid_pred, y_valid, threshold = threshold_best)
	# acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
	# print("\nStats for predictions on validation set:")
	# print(f"Threshold = {threshold_best}")
	# print(f"Accuracy = {acc}")
	# print(f"Precision = {prec}")
	# print(f"Sensitivity = {sens}")
	# print(f"Specificity = {spec}")
	# print(f"F1 score = {F1}")

if __name__ == "__main__":
    main()

