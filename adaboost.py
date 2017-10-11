import numpy as np
import argparse
from sklearn.tree import DecisionTreeClassifier

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    parser.add_argument('--numTrees', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args



def adaboost(X, y, num_iter):
    """Given an numpy matrix X, an array y and num_iter return trees and weights

    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is in {-1, 1}^n
    """
    trees = []
    trees_weights = []

    # initialize weights
    weights = [1.0 / len(y) for element in y]

    for num in range(0, num_iter):
        # fit classifier
        stumpy = DecisionTreeClassifier(max_depth=1)
        stumpy.fit(X, y, sample_weight=weights)
        predictions = stumpy.predict(X)
        trees.append(stumpy)

        # compute error
        pred_true = zip(predictions, y)
        comparison = [1.0 if pred != true else 0.0 for pred, true in pred_true]
        comp_n_weights = zip(comparison, weights)


        numerator_error = [comp * w for comp, w in comp_n_weights]
        denominator_error = np.sum([weights])
        error = np.sum(numerator_error) / denominator_error

        # compute alpha
        alpha = np.log((1 - error)/error)
        trees_weights.append(alpha)

        my_exp = [alpha * c for c in comparison]
        exp_n_weights = zip(my_exp, weights)
        weights = [w * np.exp(e) for e, w in exp_n_weights]

    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y

    assume Y in {-1, 1}^n
    """
    Yh = []
    for i in range(len(trees_weights)):
        Y = trees[i].predict(X)
        Y = np.array(Y)
        Yh.append(Y)

    Yh = np.dot(trees_weights, Yh)
    Yhat = [1.0 if y > 0 else -1.0 for y in Yh]

    return Yhat



def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row.
    """

    data = np.loadtxt(filename, delimiter=',')
    my_row, my_col = data.shape
    Y = data[:, my_col - 1]
    X = data[:, :my_col - 1]
    return X, Y



def new_label(Y):
    """ Transforms a vector of 0s and 1s to -1s and 1s.
    """
    return [-1. if y == 0. else 1. for y in Y]



def old_label(Y):
    return [0. if y == -1. else 1. for y in Y]


def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))


def main():
    """
    This code is called from the command line via
    
    python adaboost.py --train [path to filename] --test [path to filename] --numTrees 
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    num_trees = int(args['numTrees'][0])
    print train_file, test_file, num_trees


    X_train, Y = parse_spambase_data(train_file)
    X_test, Y_test = parse_spambase_data(test_file)

    Y = new_label(Y)
    trees, t_weights = adaboost(X_train, Y, num_trees)
    Yhat = adaboost_predict(X_train, trees, t_weights)
    Yhat_test = adaboost_predict(X_test, trees, t_weights)

    Yhat = old_label(Yhat)
    Yhat_test = old_label(Yhat_test)

    total = np.loadtxt(test_file, delimiter=',')
    all_total = np.column_stack((total, Yhat_test))
    np.savetxt('predictions.txt', all_total, fmt='%.6g', delimiter=',')

    Y = np.array(old_label(Y))

    # here print accuracy and write predictions to a file
    acc = accuracy(Y, Yhat)
    acc_test = accuracy(Y_test, Yhat_test)

    print("Train Accuracy %.4f" % acc)
    print("Test Accuracy %.4f" % acc_test)

if __name__ == '__main__':
    main()

