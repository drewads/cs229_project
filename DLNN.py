import evaluate
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import regularizers
# from keras.utils.vis_utils import plot_model
import img_proc
from pathlib import Path
import argparse

def DLNN(data_gen,nn_dims,epochs = 500,batch_size = 25,loss='binary_crossentropy',lambd = 1e-4):
	""" 
	Inputs:
		 data_gen - img_proc.Data_Generator that produces (x, y) training data in batches
		 nn_dims - list with number of neurons in each layer [#input layer, # 1st hidden layer, ..., 1 neuron with sigmoid]
		 epochs - number of iterations to run gradient descent 
		 batch_size - number of training examples to include in a batch
		 loss - loss function for gradient descent to optimize over (input as string)

	Outputs:
		trained_model - model trained in keras

	"""
	n = data_gen.num_features_flat()
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
	trained_model.fit(data_gen, epochs = epochs, use_multiprocessing=False, workers=8)

	return trained_model

def main(data_dir):
	BATCH_SIZE = 100
	data_gen_train = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=True, flatten=True)
	model = DLNN(data_gen_train,[1024,256,64,16,4,1],epochs = 20)
	model.save('savedDNN_' + str(data_dir))

	data_gen_train_test = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=False, flatten=True)
	y_train = data_gen_train_test.get_labels()
	y_train_pred = model.predict(data_gen_train_test)

	data_gen_valid = img_proc.Data_Generator(data_dir / 'valid', BATCH_SIZE, shuffle=False, flatten=True)
	y_valid = data_gen_valid.get_labels()
	y_valid_pred = model.predict(data_gen_valid)
    
	auc_roc,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_dlnn.jpeg')

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
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_dir', default='data')
	args = parser.parse_args()

	data_dir = Path(args.data_dir)
	main(data_dir)

