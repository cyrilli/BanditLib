import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
    def __init__(self, id, theta = None, CoTheta = None):
        self.id = id
        self.theta = theta

class UserManager():
    def __init__(self, dimension, userNum, thetaFunc, argv = None):
        self.dimension = dimension
        self.thetaFunc = thetaFunc
        self.userNum = userNum
        self.argv = argv

    def simulateThetaForHomoUsers(self):
        users = []
        thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
        l2_norm = np.linalg.norm(thetaVector, ord=2)
        thetaVector = thetaVector/l2_norm
        for key in range(self.userNum):
            users.append(User(key, thetaVector))

        return users

    def simulateThetaForHeteroUsers(self, global_dim):
        local_dim = self.dimension-global_dim
        users = []
        thetaVector_g = self.thetaFunc(global_dim, argv=self.argv)
        l2_norm = np.linalg.norm(thetaVector_g, ord=2)
        thetaVector_g = thetaVector_g/l2_norm
        for key in range(self.userNum):
            thetaVector_l = self.thetaFunc(local_dim, argv=self.argv)
            l2_norm = np.linalg.norm(thetaVector_l, ord=2)
            thetaVector_l = thetaVector_l/l2_norm

            thetaVector = np.concatenate([thetaVector_g, thetaVector_l])
            users.append(User(key, thetaVector))

        return users

class Article():	
    def __init__(self, aid, FV=None):
        self.id = aid
        self.featureVector = FV
        
class ArticleManager():
    def __init__(self, dimension, n_articles ):
        self.dimension = dimension
        self.n_articles = n_articles

    def simulateArticlePool(self):
        articles = []

        # standard d-dimensional basis (with a bias term)
        basis = np.eye(self.dimension)
        basis[:, -1] = 1
        # arm features in a unit (d - 2)-sphere
        X = np.random.randn(self.n_articles, self.dimension - 1)
        X /= np.sqrt(np.square(X).sum(axis=1))[:, np.newaxis]
        X = np.hstack((X, np.ones((self.n_articles, 1))))  # bias term
        X[: basis.shape[0], :] = basis
        for key in range(self.n_articles):
            featureVector = X[key]
            l2_norm = np.linalg.norm(featureVector, ord =2)
            articles.append(Article(key, featureVector/l2_norm ))

        return articles