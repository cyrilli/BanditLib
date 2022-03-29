# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:00:54 2020

@author: cl5ev
"""
from bidict import bidict
import random
import numpy as np

dataset = "lastfm"
data_path = "/u/cl5ev/server_jobs/Dataset/hetrec2011-{}-2k/raw_data/".format(dataset)
save_path = "/u/cl5ev/server_jobs/Dataset/hetrec2011-{}-2k/processed_data/".format(dataset)
userObservations_filename = data_path+"processed_events.dat"

user2ItemSeqs = {}

with open(userObservations_filename) as f:
    n=0
    for line in f:
        if n > 0:
            split_list=line.split('\t')
            userID = split_list[0]
            timeStamp = split_list[1]
            itemList = split_list[2]
            #print(itemList)

            if int(userID) not in user2ItemSeqs:
                user2ItemSeqs[int(userID)] = []
            user2ItemSeqs[int(userID)].append(timeStamp+"\t"+itemList)
        else:
            first_line = line
        n += 1

threshold_len = 0
user2ItemSeqs_100 = {}
for userID in user2ItemSeqs.keys():
    if len(user2ItemSeqs[userID]) >= threshold_len:
        user2ItemSeqs_100[userID] = user2ItemSeqs[userID]
print('user number: '+str(len(user2ItemSeqs_100)))
# file = open(save_path+"randUserOrderedTime_N{}_ObsMoreThan{}.dat".format(len(user2ItemSeqs), threshold_len),"w")
# file.write(first_line)
global_time = 0
while user2ItemSeqs_100:
    userID = random.choice(list(user2ItemSeqs_100.keys()))
    l = user2ItemSeqs_100[userID].pop(0)
    global_time += 1

    # file.write(str(userID)+"\t"+l)
    if not user2ItemSeqs_100[userID]:
        del user2ItemSeqs_100[userID]
# file.close()

print("global_time {}".format(global_time))