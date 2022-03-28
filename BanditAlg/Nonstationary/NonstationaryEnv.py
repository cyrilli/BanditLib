import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
    def __init__(self, id, theta = None):
        self.id = id
        self.theta = theta

class UserManager():
    def __init__(self, dimension, userNum, thetaFunc, gamma=None, UserGroups=1, argv = None):
        self.dimension = dimension
        self.thetaFunc = thetaFunc
        self.userNum = userNum
        self.gamma = gamma
        self.UserGroups = UserGroups
        self.argv = argv

    def simulateThetaForClusteredUsers(self):
        users = []
        # Generate a global unique parameter set
        global_parameter_set = []
        for i in range(self.UserGroups):
            thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
            l2_norm = np.linalg.norm(thetaVector, ord=2)
            new_theta = thetaVector / l2_norm

            if global_parameter_set == []:
                global_parameter_set.append(new_theta)
            else:
                dist_to_all_existing_big = all([np.linalg.norm(new_theta - existing_theta) >= self.gamma for existing_theta in global_parameter_set])
                while (not dist_to_all_existing_big):
                    thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
                    l2_norm = np.linalg.norm(thetaVector, ord=2)
                    new_theta = thetaVector / l2_norm
                    dist_to_all_existing_big = all(
                        [np.linalg.norm(new_theta - existing_theta) >= self.gamma for existing_theta in
                         global_parameter_set])
                global_parameter_set.append(new_theta)
        global_parameter_set = np.array(global_parameter_set)
        assert global_parameter_set.shape == (self.UserGroups, self.dimension)
        # Uniformly sample a parameter for each user as initial parameter
        parameter_index_for_users = np.random.randint(self.UserGroups, size=self.userNum)
        print(parameter_index_for_users)

        for key in range(self.userNum):
            parameter_index = parameter_index_for_users[key]
            users.append(User(key, global_parameter_set[parameter_index]))
            assert users[key].id == key
            assert np.linalg.norm(global_parameter_set[parameter_index] - users[key].theta) <= 0.001

        return users, global_parameter_set, parameter_index_for_users

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