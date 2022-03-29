from collections import Counter
from math import log
import numpy as np 
from random import *
from sklearn.decomposition import PCA
from itertools import chain

def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))

def pca_articles(articles, order):
    X = []
    for i, article in enumerate(articles):
        X.append(article.featureVector)
    pca = PCA()
    X_new = pca.fit_transform(X)
    # X_new = np.asarray(X)
    print('pca variance in each dim:', pca.explained_variance_ratio_) 

    print(X_new)
    #default is descending order, where the latend features use least informative dimensions.
    if order == 'random':
        np.random.shuffle(X_new.T)
    elif order == 'ascend':
        X_new = np.fliplr(X_new)
    elif order == 'origin':
        X_new = X
    for i, article in enumerate(articles):
        articles[i].featureVector = X_new[i]
    return

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class FileExists(Exception):
    def __init__(self, filename):
        self.value = filename
    def __str__(self):
        string = """Comment Out! Exception: %s already exists, pass 'force=True' as argument to overwrite file or delete the file."""%(self.value)
        return repr(string)

def getPoolArticleArr(pool_articles):
	article_arr = []
	for x in pool_articles:
		article_arr.append(np.array(x.featureVector))
	return np.array(article_arr)
	
def gaussianFeature(dimension, argv):
	mean = argv['mean'] if 'mean' in argv else 0
	std = argv['std'] if 'std' in argv else 1

	mean_vector = np.ones(dimension)*mean
	stdev = np.identity(dimension)*std
	vector = np.random.multivariate_normal(np.zeros(dimension), stdev)

	l2_norm = np.linalg.norm(vector, ord = 2)
	if 'l2_limit' in argv and l2_norm > argv['l2_limit']:
		"This makes it uniform over the circular range"
		vector = (vector / l2_norm)
		vector = vector * (random())
		vector = vector * argv['l2_limit']

	if mean != 0:
		vector = vector + mean_vector

	vectorNormalized = []
	for i in range(len(vector)):
		vectorNormalized.append(vector[i]/sum(vector))
	return vectorNormalized
	#return vector

def featureUniform(dimension, argv = None):
	vector = np.array([random() for _ in range(dimension)])

	l2_norm = np.linalg.norm(vector, ord =2)
	
	vector = vector/l2_norm
	return vector

def getBatchStats(arr):
	return np.concatenate((np.array([arr[0]]), np.diff(arr)))

def checkFileExists(filename):
	try:
		with open(filename, 'r'):
			return 1
	except IOError:
		return 0 

def fileOverWriteWarning(filename, force):
	if checkFileExists(filename):
		if force == True:
			print("Warning : fileOverWriteWarning %s"%(filename))
		else:
			raise FileExists(filename)


def vectorize(M):
	# temp = []
	# for i in range(M.shape[0]*M.shape[1]):
	# 	temp.append(M.T.item(i))
	# V = np.asarray(temp)
	# return V
	return np.reshape(M.T, M.shape[0]*M.shape[1])

def matrixize(V, C_dimension):
	# temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	# for i in range(len(V)/C_dimension):
	# 	temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	# W = temp
	# return W
	#To-do: use numpy built-in function reshape.
	return np.transpose(np.reshape(V, ( int(len(V)/C_dimension), C_dimension)))
