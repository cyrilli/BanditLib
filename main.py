import yaml
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dataset_utils import DataLoader
from Algorithms.util_functions import dict_union
from Algorithms.Standard.LinUCB import LinUCB

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    # parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, hLinUCB, factorUCB, LinUCB, etc.')
    parser.add_argument('-alg', nargs='+', help='<Required> Specify at least one algorithm to run, e.g. LinUCB, CoLin, hLinUCB, etc.', required=True)
    parser.add_argument('-config', dest='config', help='yaml config file')
    parser.add_argument('--dataset', default='LastFM', dest='dataset', help='dataset')
    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    common_params = cfg['common']

    ## Instantiate Bandit Algorithms to Run ##
    algorithms = {}
    for banditAlgName in args.alg:
        banditAlgParams = cfg[banditAlgName]
        # alternative way to instantiate algorithm via string
        # module = importlib.import_module('BanditAlg.Standard.LinUCB')
        # banditAlg = getattr(module, banditAlgName)
        banditAlg = globals()[banditAlgName]
        param_dict = dict_union(common_params, banditAlgParams)
        print("Instantiated {} with parameters \n {}".format(banditAlgName, param_dict))
        algorithms[banditAlgName] = banditAlg.construct_from_param_dict(param_dict=param_dict)

    print(algorithms[banditAlgName])

    ## Load Dataset ##
    # dataset_runner = DataLoader(dataset=str(args.dataset),
    #                             plot=True,
    #                             Write_to_File=True)
    
    ## Run experiment ##
    # startTime = datetime.datetime.now()
    # dataset_runner.runAlgorithms(algorithms, startTime)