# Adaboost
This is an implementation of adaboost using sklearn's <a href="http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">DecisionTreeClassifier</a>. 

Adaptive Boosting is an example of an <a href="https://en.wikipedia.org/wiki/Ensemble_learning">ensemble method</a> and uses the combined output of small, weak learning stumps to become a strong learner. Each misclassification is weighted to adapt to perform slightly better than the previous iteration.

The file can be called from the command line. The implementation will print the names input files as well as the test and train accuracy.

