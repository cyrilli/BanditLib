import pickle # Save model 
import matplotlib.pyplot as plt
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter
import numpy as np
from scipy.sparse import csgraph
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
from BanditAlg.conf import *

class Article():
    def __init__(self, aid, FV=None):
        self.article_id = aid
        self.contextFeatureVector = FV
        self.featureVector = FV


class DataLoader(object):
    def __init__(self, namelabel, dataset, context_dimension, batchSize=1, plot=True, Write_to_File=False):

        self.namelabel = namelabel
        assert dataset in ["LastFM", "Delicious", "MovieLens"]
        self.dataset = dataset
        self.context_dimension = context_dimension
        self.Plot = plot
        self.Write_to_File = Write_to_File
        self.batchSize = batchSize
        if self.dataset == 'LastFM':
            self.relationFileName = LastFM_relationFileName
            self.address = LastFM_address
            self.save_address = LastFM_save_address
            FeatureVectorsFileName = LastFM_FeatureVectorsFileName
            self.event_fileName = self.address + "/processed_events.dat"
        elif self.dataset == 'Delicious':
            # self.relationFileName = Delicious_relationFileName
            self.address = Delicious_address
            self.save_address = Delicious_save_address
            FeatureVectorsFileName = Delicious_FeatureVectorsFileName
            self.event_fileName = self.address + "/processed_events.dat"
        else:
            self.relationFileName = MovieLens_relationFileName
            self.address = MovieLens_address
            self.save_address = MovieLens_save_address
            FeatureVectorsFileName = MovieLens_FeatureVectorsFileName
            self.event_fileName = self.address + "/processed_events.dat"
            # Read Feature Vectors from File
        self.FeatureVectors = readFeatureVectorFile(FeatureVectorsFileName)
        self.articlePool = []

    def batchRecord(self, iter_):
        print("Iteration %d" % iter_, "Pool", len(self.articlePool), " Elapsed time",
              datetime.datetime.now() - self.startTime)

    def runAlgorithms(self, algorithms, startTime):
        self.startTime = startTime
        timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S')

        filenameWriteReward = os.path.join(self.save_address, 'AccReward' + str(self.namelabel) + timeRun + '.csv')

        end_num = 0
        while os.path.exists(filenameWriteReward):
            filenameWriteReward = os.path.join(self.save_address,'AccReward' + str(self.namelabel) + timeRun + str(end_num) + '.csv')
            end_num += 1

        filenameWriteCommCost = os.path.join(self.save_address, 'AccCommCost' + str(self.namelabel) + timeRun + '.csv')
        end_num = 0
        while os.path.exists(filenameWriteCommCost):
            filenameWriteCommCost = os.path.join(self.save_address,'AccCommCost' + str(self.namelabel) + timeRun + str(end_num) + '.csv')
            end_num += 1
        tim_ = []
        AlgReward = {}
        BatchCumlateReward = {}
        CommCostList = {}
        AlgReward["random"] = []
        BatchCumlateReward["random"] = []
        for alg_name, alg in algorithms.items():
            AlgReward[alg_name] = []
            BatchCumlateReward[alg_name] = []
            CommCostList[alg_name] = []

        if self.Write_to_File:
            with open(filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

            with open(filenameWriteCommCost, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

        userIDSet = set()
        with open(self.event_fileName, 'r') as f:
            f.readline()
            iter_ = 0
            for _, line in enumerate(f, 1):
                userID, _, pool_articles = parseLine(line)
                if userID not in userIDSet:
                    userIDSet.add(userID)
                # ground truth chosen article
                article_id_chosen = int(pool_articles[0])
                # Construct arm pool
                self.article_pool = []
                for article in pool_articles:
                    article_id = int(article.strip(']'))
                    article_featureVector = self.FeatureVectors[article_id]
                    article_featureVector = np.array(article_featureVector, dtype=float)
                    assert type(article_featureVector) == np.ndarray
                    assert article_featureVector.shape == (self.context_dimension,)
                    self.article_pool.append(Article(article_id, article_featureVector))

                # Random strategy
                RandomPicked = choice(self.article_pool)
                if RandomPicked.article_id == article_id_chosen:
                    reward = 1
                else:
                    reward = 0  # avoid division by zero
                AlgReward["random"].append(reward)

                for alg_name, alg in algorithms.items():
                    # Observe the candiate arm pool and algoirhtm makes a decision
                    if "AsyncLinUCB-AM" in alg_name:
                        pickedArticle = alg.decide_realData(self.article_pool, userID)
                    else:
                        pickedArticle = alg.decide(self.article_pool, userID)

                    # Get the feedback by looking at whether the selected arm by alg is the same as that of ground truth
                    if pickedArticle.article_id == article_id_chosen:
                        reward = 1
                    else:
                        reward = 0
                    # The feedback/observation will be fed to the algorithm to further update the algorithm's model estimation
                    if "AsyncLinUCB-AM" in alg_name:
                        alg.updateParameters_realData(pickedArticle, reward, userID)
                    else:
                        alg.updateParameters(pickedArticle, reward, userID)
                    # Record the reward
                    AlgReward[alg_name].append(reward)
                    CommCostList[alg_name].append(alg.totalCommCost)

                if iter_ % self.batchSize == 0:
                    self.batchRecord(iter_)
                    tim_.append(iter_)
                    BatchCumlateReward["random"].append(sum(AlgReward["random"]))
                    for alg_name in algorithms.keys():
                        BatchCumlateReward[alg_name].append(sum(AlgReward[alg_name]))

                    if self.Write_to_File:
                        with open(filenameWriteReward, 'a+') as f:
                            f.write(str(iter_))
                            f.write(',' + ','.join([str(BatchCumlateReward[alg_name][-1]) for alg_name in
                                                    list(algorithms.keys()) + ["random"]]))
                            f.write('\n')
                        with open(filenameWriteCommCost, 'a+') as f:
                            f.write(str(iter_))
                            f.write(',' + ','.join([str(CommCostList[alg_name][-1]) for alg_name in algorithms.keys()]))
                            f.write('\n')
                iter_ += 1

        if self.Plot:  # only plot
            linestyles = ['o-', 's-', '*-', '>-', '<-', 'g-', '.-', 'o-', 's-', '*-']
            markerlist = ['.', ',', 'o', 's', '*', 'v', '>', '<']

            # # plot the results
            fig, axa = plt.subplots(2, 1, sharex='all')
            # Remove horizontal space between axes
            fig.subplots_adjust(hspace=0)
            # fig.suptitle('Accumulated Regret and Communication Cost')
            # f, axa = plt.subplots(1)
            print("=====reward=====")
            count = 0
            for alg_name, alg in algorithms.items():
                labelName = alg_name
                axa[0].plot(tim_, [x / (y + 1) for x, y in zip(BatchCumlateReward[alg_name], BatchCumlateReward["random"])],
                        linewidth=1, marker=markerlist[count], markevery=2000, label=labelName)
                count += 1
            axa[0].legend(loc='upper left', prop={'size': 9})
            axa[0].set_xlabel("Iteration")
            axa[0].set_ylabel("Normalized reward")

            print("=====Comm Cost=====")
            count = 0
            for alg_name, alg in algorithms.items():
                labelName = alg_name
                axa[1].plot(tim_, CommCostList[alg_name], linewidth=1, marker=markerlist[count], markevery=2000, label=labelName)
                count += 1
            # axa[1].legend(loc='upper left',prop={'size':9})
            axa[1].set_xlabel("Iteration")
            axa[1].set_ylabel("Communication Cost")
            plt_path = os.path.join(self.save_address, str(self.namelabel) + str(timeRun) + '.png')
            # plt.savefig(plt_path, dpi=300,bbox_inches='tight', pad_inches=0.0)
            plt.show()

        for alg_name in algorithms.keys():
            print('%s: %.2f' % (alg_name, BatchCumlateReward[alg_name][-1]))

        return

def generateUserFeature(W):
    svd = TruncatedSVD(n_components=25)
    result = svd.fit(W).transform(W)
    return result

def vectorize(M):
	temp = []
	for i in range(M.shape[0]*M.shape[1]):
		temp.append(M.T.item(i))
	V = np.asarray(temp)
	return V

def matrixize(V, C_dimension):
	temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	for i in range(len(V)/C_dimension):
		temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	W = temp
	return W

def readFeatureVectorFile(FeatureVectorsFileName):
    FeatureVectors = {}
    with open(FeatureVectorsFileName, 'r') as f:        
        f.readline()
        for line in f:
            line = line.split("\t")            
            vec = line[1].strip('[]').strip('\n').split(';')
            FeatureVectors[int(line[0])] = np.array(vec)
    return FeatureVectors

# This code simply reads one line from the source files of Yahoo!
def parseLine(line):
        userID, tim, pool_articles = line.split("\t")
        userID, tim = int(userID), int(tim)
        pool_articles = np.array(pool_articles.strip('[').strip(']').strip('\n').split(','))
        #print pool_articles
      
        '''
        tim, articleID, click = line[0].strip().split("")
        tim, articleID, click = int(tim), int(articleID), int(click)
        user_features = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])
        
        pool_articles = [l.strip().split(" ") for l in line[2:]]
        pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
        '''
        return userID, tim, pool_articles

def save_to_file(fileNameWrite, recordedStats, tim):
    with open(fileNameWrite, 'a+') as f:
        f.write('data') # the observation line starts with data;
        f.write(',' + str(tim))
        f.write(',' + ';'.join([str(x) for x in recordedStats]))
        f.write('\n')