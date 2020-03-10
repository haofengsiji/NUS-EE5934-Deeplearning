"""
K Nearest Neighbour
"""

import numpy as np
from scipy import stats


class KNN(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:     WRITE CODE FOR THE FOLLOWING                                #
        # Compute the L2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # a distance matrix.                                                    #
        #                                                                       #
        #  Implement this function using only basic array operations and        #
        #  NOT using functions from scipy: (scipy.spatial.distance.cdist).      #
        #                                                                       #
        #########################################################################
        Xtrain_square = np.repeat(np.sum(self.X_train**2,axis=1,keepdims=True),num_test,axis=1).T # (num_test)X(num_train)
        Xtest_square = np.repeat(np.sum(X**2,axis=1,keepdims=True),num_train,axis=1) # (num_test)X(num_train)
        A = 2*np.matmul(X,self.X_train.T) # (num_test)X(num_train)
        dists = np.sqrt(Xtest_square + Xtrain_square - A) # (num_test)X(num_train)
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        y_pred = np.zeros(num_test)
        #########################################################################
        # TODO:     WRITE CODE FOR THE FOLLOWING                                #
        # Use the distance matrix to find the k nearest neighbors for each      #
        # testing points. Break ties by choosing the smaller label.             #
        #                                                                       #
        # Try to implement it without using loops (or list comprehension).      #
        #                                                                       #
        #########################################################################
        sort_index = np.argsort(dists,axis=1) # decreasing
        y_cands = self.y_train[sort_index[:]]
        y_pred = np.argmax(np.apply_along_axis(lambda x: np.bincount(x, minlength=10),1,y_cands[:,:k]),axis=1)
        #########################################################################
        #                           END OF YOUR CODE                            #
        #########################################################################

        return y_pred


