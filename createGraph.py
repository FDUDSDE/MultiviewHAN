import dgl
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from io import open
import unicodedata
import string
import re
import random
from numpy import *
import time
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 1000

def build_heterograph(funcname):
    if os.path.exists("data/node+edge/" + funcname + "-edge.csv"):
        edgename = "data/node+edge/" + funcname + "-edge.csv"
        nodename = "data/node+edge/" + funcname + "-node.csv"
        edgedata = pd.read_csv(edgename, error_bad_lines=False, warn_bad_lines=False, encoding="utf8")
        nodedata = pd.read_csv(nodename, error_bad_lines=False, warn_bad_lines=False, encoding="utf8")
        beginnum = nodedata["number"][0]
        edgedata["begin"] = edgedata["begin"] - beginnum
        edgedata["end"] = edgedata["end"] - beginnum

        src1 = []
        dst1 = []
        src2 = []
        dst2 = []
        src3 = []
        dst3 = []
        src4 = []
        dst4 = []
        for num in range(0, edgedata.shape[0]):
            if edgedata["type"][num] == "AST":
                src1.append(edgedata["begin"][num])
                dst1.append(edgedata["end"][num])
            elif edgedata["type"][num] == "CFG":
                src2.append(edgedata["begin"][num])
                dst2.append(edgedata["end"][num])
            elif edgedata["type"][num] == "DFG":
                src3.append(edgedata["begin"][num])
                dst3.append(edgedata["end"][num])
            else:
                src4.append(edgedata["begin"][num])
                dst4.append(edgedata["end"][num])

        src5 = np.array(src1)
        dst5 = np.array(dst1)
        src6 = np.array(src2)
        dst6 = np.array(dst2)
        src7 = np.array(src3)
        dst7 = np.array(dst3)
        src8 = np.array(src4)
        dst8 = np.array(dst4)

        data_dict = {}
        if len(src5):
            data_dict[('node', 'AST', 'node')] = ((src5), (dst5))
        if len(src6):
            data_dict[('node', 'CFG', 'node')] = ((src6), (dst6))
        if len(src7):
            data_dict[('node', 'DFG', 'node')] = ((src7), (dst7))
        if len(src8):
            data_dict[('node', 'PDG', 'node')] = ((src8), (dst8))

        return dgl.heterograph(data_dict)

# 节点特征初始化
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# 读取数据
def readLangs1(lang1, docname, reverse=False):
    print("Reading lines...")
    nodedata = pd.read_csv(docname, error_bad_lines=False, warn_bad_lines=False, encoding="utf8")
    pairs = nodedata["first"]
    input_lang = Lang(lang1)
    return input_lang, pairs

def readLangs2(lang1, docname, reverse=False):
    print("Reading lines...")
    nodedata = pd.read_csv(docname, error_bad_lines=False, warn_bad_lines=False, encoding="utf8")
    pairs = nodedata["next"]
    input_lang = Lang(lang1)
    return input_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData1(lang1, docname, reverse=False):
    input_lang, pairs = readLangs1(lang1, docname, reverse)
    print("Read %s sentence" % len(pairs))
    print("Counting words...")
    for num in range(0, len(pairs)):
        input_lang.addSentence(pairs[num])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    return input_lang, pairs

def prepareData2(lang1, docname, reverse=False):
    input_lang, pairs = readLangs2(lang1, docname, reverse)
    print("Read %s sentence" % len(pairs))
    print("Counting words...")
    for num in range(0, len(pairs)):
        input_lang.addSentence(str(pairs[num]))
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)

    return input_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair)

    return (input_tensor)

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)  
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=True) 

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)  # BiLSTM

# 编码
def LSTMtrain(input_tensor,
              encoder,
              num,
              max_length=MAX_LENGTH):
    encoder_hidden = (encoder.initHidden(), encoder.initHidden()) 

    input_length = input_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, 2 * encoder.hidden_size, device=device)  # BiLSTM

    for ei in range(0, input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

        result = encoder_outputs[ei].detach().cpu().numpy()

        return result

def create_feature1(funcname):
    nodename = "data/node+edge/" + funcname + "-node.csv"
    input_lang, pairs = prepareData1('code', nodename, False)
    nodedata = pd.read_csv(nodename, error_bad_lines=False, warn_bad_lines=False, encoding="utf8")

    hidden_size = 256
    n_iters = 75000
    encoder1 = EncoderLSTM(input_lang.n_words, hidden_size).to(device)

    results = []

    for num in range(0, len(pairs)):
        training_pairs = [tensorsFromPair(input_lang, pairs[num])
                          for i in range(n_iters)]  # size: n_iters X 2
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
        result = LSTMtrain(input_tensor,
                           encoder1,
                           num,
                           max_length=MAX_LENGTH)
        # 0:参数类，1：操作类
        if nodedata['type'][num] == 0:
            result1 = [n * 0.5 for n in result]
            results.append(result1)
        else:
            result1 = [n * 0.7 for n in result]
            results.append(result1)
    return results

def create_feature2(funcname):
    nodename = "data/node+edge/" + funcname + "-node.csv"
    input_lang, pairs = prepareData2('code', nodename, False)
    nodedata = pd.read_csv(nodename, error_bad_lines=False, warn_bad_lines=False, encoding="utf8")

    hidden_size = 256
    n_iters = 75000
    encoder1 = EncoderLSTM(input_lang.n_words, hidden_size).to(device)

    results = []
    for num in range(0, len(pairs)):
        training_pairs = [tensorsFromPair(input_lang, str(pairs[num]))
                          for i in range(n_iters)]  # size: n_iters X 2
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
        result = LSTMtrain(input_tensor,
                           encoder1,
                           num,
                           max_length=MAX_LENGTH)
        # 0:参数类，1：操作类
        if nodedata['type'][num] == 0:
            result1 = [n * 0.5 for n in result]
            results.append(result1)
        else:
            result1 = [n * 0.3 for n in result]
            results.append(result1)
    return results

# 边的特征初始化
def create_edge_feat(embed, G):
    if 'AST' in G.etypes:
        a = np.array(embed.weight[0].detach().numpy())
        b = tile(a, (G['AST'].number_of_edges(), 1))
        c = torch.tensor(b)
        c.reshape(G['AST'].number_of_edges(), 512).shape
        G['AST'].edata['weight'] = c

    if 'CFG' in G.etypes:
        a2 = np.array(embed.weight[1].detach().numpy())
        b2 = tile(a2, (G['CFG'].number_of_edges(), 1))
        c2 = torch.tensor(b2)
        c2.reshape(G['CFG'].number_of_edges(), 512).shape
        G['CFG'].edata['weight'] = c2

    if 'PDG' in G.etypes:
        a3 = np.array(embed.weight[2].detach().numpy())
        b3 = tile(a3, (G['PDG'].number_of_edges(), 1))
        c3 = torch.tensor(b3)
        c3.reshape(G['PDG'].number_of_edges(), 512).shape
        G['PDG'].edata['weight'] = c3

    if 'DFG' in G.etypes:
        a4 = np.array(embed.weight[3].detach().numpy())
        b4 = tile(a4, (G['DFG'].number_of_edges(), 1))
        c4 = torch.tensor(b4)
        c4.reshape(G['DFG'].number_of_edges(), 512).shape
        G['DFG'].edata['weight'] = c4


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def create_graph(funcname, embed):
    # G = build_heterograph(funcname)
    # result1 = create_feature1(funcname)
    # result2 = create_feature2(funcname)
    # G.ndata['feat'] = torch.tensor(result1) + torch.tensor(result2)
    # create_edge_feat(embed, G)
    # return G
    G = build_heterograph(funcname)
    embed1 = nn.Embedding(G.number_of_nodes(), 512)
    G.ndata['feat'] = embed1.weight
    create_edge_feat(embed, G)
    return G

def create_graphlist(funcnames, embed):
    Gs = []
    numbers = funcnames['function_name'].size
    for num in range(0, numbers):
        start = time.time()
        G = create_graph(str(funcnames['id'][num]), embed)
        Gs.append(G)
        if (num % 1000) == 0:
            print('create ', num, ' graph')
    return Gs


