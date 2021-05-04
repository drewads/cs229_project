import numpy as np
import matplotlib.pyplot as plt

def counts(y_pred, y_true, threshold = 0.5):
	""" 
	Inputs:
		 y_true - true labels (np.array(rows: #examples; cols: 1))
		 y_pred - predicted labels (np.array(rows: #examples; cols: 1))
		 threshold - decision threshold (float)

	Outputs:
		tp - true positives
		tn - true negatives
		fp - false positives
		fn - false negatives

	"""

	tp = 0
	fp = 0
	tn = 0
	fn = 0
	m = np.shape(y_true)[0]
	for i in range(m):
		if y_pred[i] >= threshold:
			if y_true[i] == 1:
				tp += 1
			else:
				fn += 1
		else:
			if y_true [i] == 1:
				fp += 1
			else:
				tn += 1

	return tp,fn,fp,tn

def stats(tp,fn,fp,tn):
	""" 
	Inputs:
		tp - true positives
		tn - true negatives
		fp - false positives
		fn - false negatives

	Outputs:
		accuracy,precision,sensitivity,specificity,F1

	"""
	eps = 1e-7
	accuracy =  (tp+tn)/(tp+fn+fp+tn)
	precision = tp/(tp+fp+eps)
	sensitivity = tp/(tp+fn+eps)
	specificity = tn/(tn+fp+eps)
	F1 = (2/((1/(sensitivity + eps)) +  (1/(precision+eps))))

	return accuracy,precision,sensitivity,specificity,F1

def ROCandAUROC(y_pred,y_true,eval_data_save_path = 'eval_data.txt',ROC_path = 'ROC.jpeg',AUROC_path = 'AUROC.jpeg'):
	length = 50
	threshold = np.linspace(0,1,length)
	acc = np.zeros((length,1))
	prec = np.zeros((length,1))
	sens = np.zeros((length,1))
	spec = np.zeros((length,1))
	F1 = np.zeros((length,1))

	for i in range(len(threshold)):
		tp,fn,fp,tn = counts(y_pred, y_true, threshold[i])
		acc[i],prec[i],sens[i],spec[i],F1[i] = stats(tp,fn,fp,tn)

	# Plot ROC Curve
	plt.figure()
	plt.step(sens, spec,where='pre')
	plt.xlabel('Sensitivity')
	plt.ylabel('Specificity')
	plt.savefig(ROC_path)

    # Plotting AUROC
	plt.figure()
	plt.step(sens,prec,where='pre')
	plt.xlabel('Precision')
	plt.ylabel('Sensitivity')
	plt.savefig(AUROC_path)

	threshold = np.reshape(threshold, (threshold.shape[0],1))
	acc = np.reshape(acc, (acc.shape[0],1))
	prec = np.reshape(prec, (prec.shape[0],1))
	sens = np.reshape(sens, (sens.shape[0],1))
	spec = np.reshape(spec, (spec.shape[0],1))
	F1 = np.reshape(F1, (F1.shape[0],1))

    #saving data
	np.savetxt(eval_data_save_path,np.concatenate((threshold,acc,prec,sens,spec,F1), axis=1), header='threshold,accuracy,precision,sensitivity,specificity,F1')


