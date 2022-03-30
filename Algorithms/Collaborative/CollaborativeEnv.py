import numpy as np 
import json
from random import randint


class User():
	def __init__(self, id, theta = None, CoTheta = None):
		self.id = id
		self.theta = theta
		self.CoTheta = CoTheta


class UserManager():
	def __init__(self, dimension, userNum,  UserGroups, thetaFunc, argv = None):
		self.dimension = dimension
		self.thetaFunc = thetaFunc
		self.userNum = userNum
		self.UserGroups = UserGroups
		self.argv = argv
		self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__

	def generateMasks(self):
		mask = {}
		for i in range(self.UserGroups):
			mask[i] = np.random.randint(2, size = self.dimension)
		return mask

	def simulateThetafromUsers(self):
		usersids = {}
		users = []
		mask = self.generateMasks()

		if (self.UserGroups == 0):
			for key in range(self.userNum):
				thetaVector = self.thetaFunc(self.dimension, argv = self.argv)
				l2_norm = np.linalg.norm(thetaVector, ord =2)
				users.append(User(key, thetaVector/l2_norm))
		else:
			for i in range(self.UserGroups):
				usersids[i] = range(round(self.userNum*i/self.UserGroups), round((self.userNum*(i+1))/self.UserGroups))

				for key in usersids[i]:
					thetaVector = np.multiply(self.thetaFunc(self.dimension, argv = self.argv), mask[i])
					l2_norm = np.linalg.norm(thetaVector, ord =2)
					users.append(User(key, thetaVector/l2_norm))
		return users

class Article():	
	def __init__(self, aid, FV=None):
		self.id = aid
		self.featureVector = FV
		
class ArticleManager():
	def __init__(self, dimension, n_articles, ArticleGroups, FeatureFunc, argv ):
		self.signature = "Article manager for simulation study"
		self.dimension = dimension
		self.n_articles = n_articles
		self.ArticleGroups = ArticleGroups
		self.FeatureFunc = FeatureFunc
		self.argv = argv
		self.signature = "A-"+str(self.n_articles)+"+AG"+ str(self.ArticleGroups)+"+TF-"+self.FeatureFunc.__name__

	def saveArticles(self, Articles, filename, force = False):
		with open(filename, 'w') as f:
			for i in range(len(Articles)):
				f.write(json.dumps((Articles[i].id, Articles[i].featureVector.tolist())) + '\n')


	def loadArticles(self, filename):
		articles = []
		with open(filename, 'r') as f:
			for line in f:
				aid, featureVector = json.loads(line)
				articles.append(Article(aid, np.array(featureVector)))
		return articles

	#automatically generate masks for articles, but it may generate same masks
	def generateMasks(self):
		mask = {}
		for i in range(self.ArticleGroups):
			mask[i] = np.random.randint(2, size = self.dimension)
		return mask

	def simulateArticlePool(self):
		articles = []
		
		articles_id = {}
		mask = self.generateMasks()
		feature_matrix = np.empty([self.n_articles, self.dimension])
		for i in range(self.dimension):
			feature_matrix[:, i] = np.random.normal(0, np.sqrt(1.0*(self.dimension-i)/self.dimension), self.n_articles)
		if (self.ArticleGroups == 0):
			for key in range(self.n_articles):
				featureVector = feature_matrix[key]
				l2_norm = np.linalg.norm(featureVector, ord =2)
				articles.append(Article(key, featureVector/l2_norm ))
		else:
			for i in range(self.ArticleGroups):
				articles_id[i] = range(round((self.n_articles*i)/self.ArticleGroups), round((self.n_articles*(i+1))/self.ArticleGroups))

				for key in articles_id[i]:
					featureVector = np.multiply(feature_matrix[key], mask[i])
					l2_norm = np.linalg.norm(featureVector, ord =2)
					articles.append(Article(key, featureVector/l2_norm ))	
		return articles