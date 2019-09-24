""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""
# TO DO - implement a prediction helper method since perceptron and mclogistic 
# have the same predictor
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
        for j in range(X.shape[0]):
            maxvalue = 0
            argmax = 0
            for i in range(self.W.shape[0]):
                pred = np.dot(self.W[i], np.ravel(X[j].todense()))
                if pred > maxvalue:
                    maxvalue = pred
                    argmax = i
            label = y[j]
            if argmax != label:
                self.W[argmax] = self.W[argmax] - lr * np.ravel(X[j].todense())
                self.W[label] = self.W[label] + lr * np.ravel(X[j].todense())
    
    def predict(self, X):
        X = self._fix_test_feats(X)
        y = np.zeros(X.shape[0], dtype=np.float)
        for j in range(X.shape[0]):
            maxvalue = 0
            argmax = 0
            for i in range(self.W.shape[0]):
                pred = np.dot(self.W[i], np.ravel(X[j].todense()))
                if pred > maxvalue:
                    maxvalue = pred
                    argmax = i
            y[j] = argmax
        return y

    def score(self, x):
        return np.dot(self.W[0], x)


class MCLogistic(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        logits = np.zeros(self.W.shape[0], dtype=np.float)
        for j in range(X.shape[0]):
            label = y[j]
            example = np.ravel(X[j].todense())
            for k in range(self.W.shape[0]):
                logits[k] = np.dot(self.W[k], example)
            softmax = self.softmax(logits)
            for k in range(self.W.shape[0]):
                if(k == label):
                    self.W[k] = self.W[k] + (lr*(example - softmax[k]*example)) 
                else:
                    self.W[k] = self.W[k] + (lr*(-1 * softmax[k]*example))
                
    def predict(self, X):
        X = self._fix_test_feats(X)
        y = np.zeros(X.shape[0], dtype=np.float)
        for j in range(X.shape[0]):
            maxvalue = 0
            argmax = 0
            for i in range(self.W.shape[0]):
                pred = np.dot(self.W[i], np.ravel(X[j].todense()))
                if pred > maxvalue:
                    maxvalue = pred
                    argmax = i
            y[j] = argmax
        return y

    def softmax(self, logits):
        logits = np.exp(logits - max(logits))
        return logits/sum(logits)

    def score(self, x):
        return np.dot(self.W[0], x)

class OneVsAll(Model):

    def __init__(self, *, nfeatures, nclasses, model_class):
        super().__init__(nfeatures)
        self.num_classes = nclasses
        self.model_class = model_class
        self.models = [model_class(nfeatures=nfeatures, nclasses=2) for _ in range(nclasses)]

    def fit(self, *, X, y, lr):
        for i in range(len(self.models)):
            model = self.models[i]
            y_i = np.zeros(len(y), dtype=np.int)
            for j in range(len(y)):
                if y[j] == i:
                    y_i[j] = 0
                else:
                    y_i[j] = 1
            model.fit(X=X, y=y_i, lr=lr)
            print(model.W)
 
    def predict(self, X):
        X = self._fix_test_feats(X)
        y = np.zeros(X.shape[0], dtype=np.float)
        for j in range(X.shape[0]):
            maxvalue = 0
            argmax = 0
            for i in range(len(self.models)):
                model = self.models[i]
                pred = model.score(np.ravel(X[j].todense()))
                if pred > maxvalue:
                    maxvalue = pred
                    argmax = i
            y[j] = argmax
        return y
