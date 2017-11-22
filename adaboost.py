"""
Updated 11/2017
Adaboost function tested on spam-email dataset.
Data from UCI Machine Learning Repository http://www.ics.uci.edu/~mlearn/MLRepository.html
Number of Attributes: 58 (57 continuous, 1 nominal class label [spam-1, not spam-0])

Train: 3600 observations
Test: 1000 obseravtions

This file can be run from command line. Train and test files are hardcoded.
Returns train and test accuracy, number of trees used, and saves predictions appended to test set in file predictions.txt.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold


def parse_spambase_data(filename):
    """ Given a filename;
    return X and Y numpy arrays

    Y is in {-1, 1}
    """
    data = np.loadtxt(filename, delimiter=',')
    n_row, n_col = data.shape
    Y = np.asarray(new_label(data[:, n_col - 1]))
    X = data[:, :n_col - 1]

    return X, Y


def adaboost(X, y, num_iter):
    """Given arrays X, y and number of trees;
    return lists of classifiers trees and corresponding tree_weights

    y is in {-1, 1}
    """
    trees = []
    trees_weights = []

    # initialize weights
    weights = [1.0 / len(y) for element in y]
    seed = 42

    for num in range(0, num_iter):
        # fit classifier
        stump = DecisionTreeClassifier(max_depth= 1, splitter= "random", random_state= seed)
        stump.fit(X, old_label(y), sample_weight= weights)
        predictions = stump.predict(X)
        trees.append(stump)
        seed += 1

        # compute error
        comparison = [0.0 if pred == true else 1.0 for pred, true in zip(new_label(predictions), y)]
        numerator_error = [comp * w for comp, w in zip(comparison, weights)]
        denominator_error = np.sum(weights)
        error = np.sum(numerator_error) / denominator_error

        # compute alpha
        alpha = np.log((1 - error)/error)
        trees_weights.append(alpha)

        # update weights
        my_exp = [alpha * c for c in comparison]
        weights = [np.exp(e) * w for e, w in zip(my_exp, weights)]

    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, lists of trees and tree_weights;
    return predictions Yhat

    Yhat is in {-1, 1}
    """
    Yh = []
    for i in range(len(trees_weights)):
        Y = new_label(trees[i].predict(X))
        Yh.append(Y)

    Yh = np.dot(trees_weights, Yh)
    Yhat = np.sign(Yh)
    return Yhat


def new_label(Y):
    """Given a vector of 0s and 1s;
    return vector of -1s and 1s
    """
    return [-1. if y == 0. else 1. for y in Y]


def old_label(Y):
    """Given a vector of -1s and 1s;
    return vector of 0s and 1s
    """
    return [0. if y == -1. else 1. for y in Y]


def accuracy(pred, y_true):
    return np.sum([p == t for p, t in zip(pred, y_true)]) / float(len(y_true))


def find_num_of_trees(file_train, tree_range):
    """Given training data file and range of number of classification trees;
    return number of trees with lowest CV error

    Yhat is in {-1, 1}
    """
    mean_CV_err = []
    X, Y = parse_spambase_data(file_train)
    print("Finding best number of trees through CV...")

    for num in tree_range:
        error_valid_list = []

        kf = KFold(3, shuffle= True, random_state= 42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            trees, t_weights = adaboost(X_train, y_train, num)
            Yhat_test = adaboost_predict(X_test, trees, t_weights)

            acc_valid = accuracy(Yhat_test, y_test)

            error_valid = 1 - acc_valid
            error_valid_list.append(error_valid)

        mean_CV_err.append(np.mean(error_valid_list))

    index = np.argmin(mean_CV_err)
    return tree_range[index], mean_CV_err


def main():
    num_trees, mean_CV_err = find_num_of_trees(file_train, tree_range)
    print("Best number of trees:", num_trees)

    X_train, Y_train = parse_spambase_data(file_train)
    X_test, Y_test = parse_spambase_data(file_test)

    trees, t_weights = adaboost(X_train, Y_train, num_trees)
    Yhat = adaboost_predict(X_train, trees, t_weights)
    Yhat_test = adaboost_predict(X_test, trees, t_weights)

    # results
    acc_train = accuracy(Yhat, Y_train)
    acc_test = accuracy(Yhat_test, Y_test)

    print("Train Accuracy %.4f" % acc_train)
    print("Test Accuracy %.4f" % acc_test)

    # saving results
    total = np.loadtxt(file_test, delimiter=',')
    all_total = np.column_stack((total, Yhat_test))
    np.savetxt(results, all_total, fmt='%.6g', delimiter=',')


###################################################################################

file_train = "email.train.txt"
file_test = "email.test.txt"
results = "predictions.txt"
tree_range = np.arange(100, 1100, 100)


if __name__ == "__main__":
    main()

