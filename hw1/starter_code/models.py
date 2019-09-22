""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures

    def fit(self, *, X, y, lr):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class MCModel(Model):
    """ A multiclass model abstraction.
    It wants to know, up front:
        - How many features in the data
        - How many classes in the data
    """

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures)
        self.num_classes = nclasses


class MCPerceptron(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        # train.mc.perceptron
        #---------------------
        # 200 rows of examples, 617 columns of features
        # X[0] is the first example, a scipy csr matrix indexed (0,0 -> (0,617)
        # y[0] is the first label(ndarray)
        # W is an ndarray 26x617, one row for each letter, 617 rows for each feature
        
        # pseudocode
        #-----------
        # for each example in X:
            # do w . x_i for each row of W and find the maximum prediction 
            # and return that k value as our predicted y-hat

            # check the actual label, if !=:
                # update the w sub k of the prediction
                # update the w sub k of the actual label
                # according to the update rules, and pop these back into W

        # Remember that features are 1-indexed, but the W matrix is 0 indexed.
        for j in range(X.shape[0]):
            maxvalue = 0
            argmax = 0
            for i in range(W.shape[0]):
                pred = np.dot(W[i], X[j])
                if pred > maxvalue:
                    maxvalue = pred
                    argmax = i
            label = y[j]
            if argmax != label:
                W[argmax] = W[argmax] - lr * X[j]
                W[label] = W[label] - lr * X[j]

    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        raise Exception("You must implement this method!")


class MCLogistic(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def softmax(self, logits):
        # TODO: Implement this!
        raise Exception("You must implement this method!")


class OneVsAll(Model):

    def __init__(self, *, nfeatures, nclasses, model_class):
        super().__init__(nfeatures)
        self.num_classes = nclasses
        self.model_class = model_class
        self.models = [model_class(nfeatures=nfeatures, nclasses=2) for _ in range(nclasses)]

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        raise Exception("You must implement this method!")
