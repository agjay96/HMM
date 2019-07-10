import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
    if len(predict_tagging) != len(true_tagging):
        return 0, 0, 0
    cnt = 0
    for i in range(len(predict_tagging)):
        if predict_tagging[i] == true_tagging[i]:
            cnt += 1
    total_correct = cnt
    total_words = len(predict_tagging)
    if total_words == 0:
        return 0, 0, 0
    return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

    def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
        tags = self.read_tags(tagfile)
        data = self.read_data(datafile)
        self.tags = tags
        lines = []
        for l in data:
            new_line = self.Line(l)
            if new_line.length > 0:
                lines.append(new_line)
        if seed is not None: random.seed(seed)
        random.shuffle(lines)
        train_size = int(train_test_split * len(data))
        self.train_data = lines[:train_size]
        self.test_data = lines[train_size:]
        return

    def read_data(self, filename):
        """Read tagged sentence data"""
        with open(filename, 'r') as f:
            sentence_lines = f.read().split("\n\n")
        return sentence_lines

    def read_tags(self, filename):
        """Read a list of word tag classes"""
        with open(filename, 'r') as f:
            tags = f.read().split("\n")
        return tags

    class Line:
        def __init__(self, line):
            words = line.split("\n")
            self.id = words[0]
            self.words = []
            self.tags = []

            for idx in range(1, len(words)):
                pair = words[idx].split("\t")
                self.words.append(pair[0])
                self.tags.append(pair[1])
            self.length = len(self.words)
            return

        def show(self):
            print(self.id)
            print(self.length)
            print(self.words)
            print(self.tags)
            return


# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    
    ###################################################
    # Edit here
    N=len(train_data)
    S=len(tags)
    pi=np.zeros(S)
    A=np.zeros((S,S))
    state_dict={}
    obs=[]
    obs_dict={}
    o=0
    for t in range(S):
        state_dict[tags[t]]=t
    
    for line in train_data:
        pi[state_dict[line.tags[0]]]+=1
        for w in range(line.length-1):
            A[state_dict[line.tags[w]],state_dict[line.tags[w+1]]]+=1

    for line in train_data:
        for w in range(line.length):
            if line.words[w] not in obs_dict.keys():
                obs_dict[line.words[w]]=o
                o+=1  
    
    pi=pi/N
    A=(A.T/np.sum(A, axis=1)).T

               
    O=len(obs_dict)
    
    B=np.zeros((S,O))
    
    for line in train_data:
        for w in range(line.length):
            B[state_dict[line.tags[w]],obs_dict[line.words[w]]]+=1
    B=(B.T/np.sum(B,axis=1)).T
    a1=np.isnan(A)
    A[a1]=0
    b1=np.isnan(B)
    B[b1]=0
    model=HMM(pi,A,B,obs_dict,state_dict)
    
    return model


# TODO:
def speech_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    # Edit here
    for line in test_data:
        for w in line.words:
            if(w not in model.obs_dict.keys()):
                L=len(model.obs_dict)
                model.obs_dict[w]=L
                b=np.ones((len(model.state_dict),1))
                b=b*1e-6
                model.B=np.append(model.B,b,1)
        tagging.append(model.viterbi(line.words))
        ###################################################
    return tagging

