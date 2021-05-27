from random import shuffle
import numpy as np
import img_proc
import evaluate

def main():
    """Problem: Logistic regression
    """
    # for step_size in [0.1,0.01]:
    #     x_train, y_train = next(img_proc.slice_data_sequential('data/train_sep', 500))
    #     clf = LogisticRegression(step_size=step_size)
    #     clf.fit(normalize(x_train), y_train)
    #
    #     x_valid, y_valid = next(img_proc.slice_data_sequential('data/valid', 500))
    #     y_train_pred = clf.predict(normalize(x_train))
    #     y_valid_pred = clf.predict(normalize(x_valid))
    #
    #     auc_roc,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_log_reg.jpeg')
    #
    #     print(f"Stats for step size {step_size}")
    #     print(f"\nArea Under ROC = {auc_roc}")
    #     tp,fn,fp,tn = evaluate.counts(y_train_pred, y_train, threshold = threshold_best)
    #     acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    #     print("\nStats for predictions on train set:")
    #     print(f"Threshold = {threshold_best}")
    #     print(f"Accuracy = {acc}")
    #     print(f"Precision = {prec}")
    #     print(f"Sensitivity = {sens}")
    #     print(f"Specificity = {spec}")
    #     print(f"F1 score = {F1}")
    #
    #     tp,fn,fp,tn = evaluate.counts(y_valid_pred, y_valid, threshold = threshold_best)
    #     acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    #     print("\nStats for predictions on validation set:")
    #     print(f"Threshold = {threshold_best}")
    #     print(f"Accuracy = {acc}")
    #     print(f"Precision = {prec}")
    #     print(f"Sensitivity = {sens}")
    #     print(f"Specificity = {spec}")
    #     print(f"F1 score = {F1}")
    step_size = 0.01
    BATCH_SIZE = 100
    data_gen_train = img_proc.Data_Generator('data/train_sep', BATCH_SIZE, shuffle=True, flatten=True)
    print("mini batch")
    clf2 = LogisticRegression()
    clf2.fit_from_gen(data_gen_train)

    data_gen_train_test = img_proc.Data_Generator('data/train_sep', BATCH_SIZE, shuffle=False, flatten=True)
    y_train = data_gen_train_test.labels()
    y_train_pred2 = clf2.predict_from_gen(data_gen_train_test)

    data_gen_valid = img_proc.Data_Generator('data/valid', BATCH_SIZE, shuffle=False, flatten=True)
    y_valid = data_gen_valid.labels()
    y_valid_pred2 = clf2.predict_from_gen(data_gen_valid)

    auc_roc, threshold_best = evaluate.ROCandAUROC(y_valid_pred2, y_valid, 'ROC_valid_data_log_reg_mini_batch.jpeg')

    print(f"Stats for step size {step_size}")
    print(f"\nArea Under ROC = {auc_roc}")
    tp, fn, fp, tn = evaluate.counts(y_train_pred2, y_train, threshold=threshold_best)
    acc, prec, sens, spec, F1 = evaluate.stats(tp, fn, fp, tn)
    print("\nStats for predictions on train set:")
    print(f"Threshold = {threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")

    tp, fn, fp, tn = evaluate.counts(y_valid_pred2, y_valid, threshold=threshold_best)
    acc, prec, sens, spec, F1 = evaluate.stats(tp, fn, fp, tn)
    print("\nStats for predictions on validation set:")
    print(f"Threshold = {threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")

class LogisticRegression:

    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def sigmoid_theta(self, x):
        return 1 / (1 + np.exp(-np.dot(self.theta, x)))

    def fit(self, x, y):
        """

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        if self.theta == None:
            self.theta = np.zeros(np.shape(x)[1])
        for i in range(self.max_iter):
            addition = np.zeros(len(x[0]))
            for j in range(len(x)):
                addition += x[j] * (y[j] - self.sigmoid_theta(x[j]))
            self.theta += self.step_size * addition
            error = np.linalg.norm(addition * self.step_size)
            if (error < self.eps):
                return

    def fit_mini_batch(self, x, y, batch_size):
        if self.theta == None:
            self.theta = np.zeros(np.shape(x)[1])

        for i in range(self.max_iter):
            for k in range(int(len(x) / batch_size)):
                addition = np.zeros(len(x[0]))
                for j in range(batch_size):
                    addition += x[k * batch_size + j] * (y[k * batch_size + j] - self.sigmoid_theta(x[k * batch_size + j]))
                self.theta += self.step_size * addition
                error = np.linalg.norm(addition * self.step_size)
                if (error < self.eps):
                    print(f"It took {i} iterations with step size {self.step_size}")
                    return

    def fit_from_gen(self, data_gen):
        if self.theta == None:
            self.theta = np.zeros(data_gen.num_features_flat())

        for epoch in range(self.max_iter):
            error_epoch = 0
            for batch in range(data_gen.__len__()):
                x_batch, y_batch = data_gen.__getitem__(batch)
                gradient = np.zeros(data_gen.num_features_flat())
                
                for i in range(x_batch.shape[0]):
                    gradient += x_batch[i] * (y_batch[i] - self.sigmoid_theta(x_batch[i]))
                self.theta += self.step_size * gradient

                error_epoch += np.linalg.norm(gradient * self.step_size)
            
            if (error_epoch / data_gen.__len__() < self.eps):
                print(f"It took {i} iterations with step size {self.step_size}")
                return
            
            data_gen.on_epoch_end()

    def predict_from_gen(self, data_gen):
        outputs = []

        for batch in range(data_gen.__len__()):
            x_batch, y_batch = data_gen.__getitem__(batch)
            predicted_probability_batch = np.exp(np.dot(x_batch, self.theta))
            outputs_batch = np.rint(predicted_probability_batch)

            for i in range(outputs_batch.shape[0]):
                outputs.append(outputs_batch[i])
        
        return np.array(outputs)

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        predicted_probability = np.exp(np.dot(x, self.theta))
        outputs = np.empty(predicted_probability.shape)
        for i in range(len(x)):
            if predicted_probability[i] >= 0.5:
                outputs[i] = 1
            else:
                outputs[i] = 0
        return outputs


if __name__ == '__main__':
    main()
