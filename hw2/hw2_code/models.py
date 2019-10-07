"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np

class RegressionTree(object):

    class Node(object):

        def __init__(self, depth):
            self.leaf = False
            self.depth = depth 
            self.left = None
            self.right = None
            self.d = 0
            self.theta = 0
            self.mean = 0
    
    def __init__(self, nfeatures, max_depth):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.root = None

    def buildTree(self, root, X, y):
        if X.shape[0] <= 1:
            root.leaf = True
            root.mean = np.mean(y)
            return

        if root.depth == self.max_depth:
            root.leaf = True
            root.mean = np.mean(y)
            return

        zero_variance = True
        for row in X.T:
            var = np.var(row)
            if var != 0:
                zero_variance = False
        if zero_variance == True:
            root.leaf = True
            root.mean = np.mean(y)
            return

        scores = {}
        for i, row in enumerate(X.T):
            if np.var(row) != 0:
                for x in row:
                    tuple_ = (i, x)        
                    if tuple_ not in scores:
                        l_index = []
                        r_index = []
                        for k, x_ in enumerate(row):
                            if x_ < x:
                                l_index.append(k)
                            else:
                                r_index.append(k)
                        if (len(l_index) > 0 and len(r_index) > 0):
                            y = np.array(y, dtype=float)
                            l_value = y[l_index]
                            r_value = y[r_index]
                            l_mean = np.mean(l_value)
                            r_mean = np.mean(r_value)
                            score = np.sum(np.square(l_value - l_mean)) + np.sum(np.square(r_value - r_mean))
                            score_tuple = (score, l_index, r_index)
                            scores.update({tuple_:score_tuple})
        tuples = list(scores.keys())
        scores_tuple = list(scores.values())
        scores_list = [i[0] for i in scores_tuple]
        index = np.argmin(scores_list)
        min_tuple = tuples[index]
        min_scores_tuple = scores_tuple[index]
        root.d = min_tuple[0]
        root.theta = min_tuple[1]
        X_left = X[min_scores_tuple[1]]
        X_right = X[min_scores_tuple[2]]
        y_left = y[min_scores_tuple[1]]
        y_right = y[min_scores_tuple[2]]
        root.left = self.Node(root.depth + 1)
        root.right = self.Node(root.depth + 1)
        self.buildTree(root.left, X_left, y_left)
        self.buildTree(root.right, X_right, y_right) 
 
    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
        """
        self.root = self.Node(0)
        self.buildTree(self.root, X, y)

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        y = []
        for row in X:
            node = self.root
            while(node.leaf != True):
                if(row[node.d] < node.theta):
                    node = node.left
                else:
                    node = node.right
            y.append(node.mean)
        return np.array(y, dtype=float)

class GradientBoostedRegressionTree(object):
    def __init__(self, nfeatures, max_depth, n_estimators, regularization_parameter):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.regularization_parameter = regularization_parameter
    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
                n_estimators: An int representing the number of regression trees to iteratively fit
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")
