import numpy as np
import img_proc
import evaluate

def normalize(X):
    m = np.shape(X)[0] # number of examples
    n = np.shape(X)[1] # number of features in an example
    mu = np.reshape(np.sum(X,axis=0),(1,n))/m
    return (X - mu)/255
    # Sigma = np.cov(X.T)
    # return np.solve(Sigma,X_centred)


def main():
    """Problem: Logistic regression

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = next(img_proc.slice_data_sequential('data/train_sep', 100))
    clf = LogisticRegression()
    clf.fit(normalize(x_train), y_train)

    x_valid, y_valid = next(img_proc.slice_data_sequential('data/valid', 100))
    y_train_pred = clf.predict(normalize(x_train))
    y_valid_pred = clf.predict(normalize(x_valid))

    auc_roc,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_log_reg.jpeg')

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
        print("error")
        for i in range(self.max_iter):
            addition = np.zeros(len(x[0]))
            for j in range(len(x)):
                #addition += x[j] * (y[j] - np.exp(np.dot(self.theta, x[j])))
                addition += x[j] * (y[j] - self.sigmoid_theta(x[j]))
            self.theta += self.step_size * addition
            error = np.linalg.norm(addition * self.step_size)
            print(error)
            if (error < self.eps):
                return


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
