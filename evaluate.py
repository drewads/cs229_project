from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
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
			if y_true[i] == 1.0:
				tp += 1
			else:
				fn += 1
		else:
			if y_true [i] == 1.0:
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
	sensitivity = tp/(tp+fn+eps) # = recall = tp/(tp+fn+eps)
	specificity = tn/(tn+fp+eps)
	F1 = (2/((1/(sensitivity + eps)) +  (1/(precision+eps))))

	return accuracy,precision,sensitivity,specificity,F1

def ROCandAUROC(y_pred,y_true,ROC_path = 'ROC.jpeg'):

	#getting true positive rate and false positive rate
	fpr, tpr, thresholds = roc_curve(y_true, y_pred)

	# Plot ROC Curve
	plt.figure()
	plt.plot([0, 1], ls="--")
	plt.plot([0, 0], [1, 0] , c=".7")
	plt.plot([1, 1] , c=".7")
	plt.plot(fpr, tpr)
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(ROC_path)

	auc = roc_auc_score(y_true, y_pred)

	#finding best threshold
	p = fpr*np.flip(tpr)
	indx_best = 0
	for i in range(len(p)):
		if p[indx_best] <= p[i]:
			indx_best = i

	return auc, thresholds[indx_best]
