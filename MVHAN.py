from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import time
import math
from dgl.nn.pytorch import GATConv
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

import createGraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 设置
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 1000

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
def readLangs(lang1, lang2, docname, reverse=False):
    print("Reading lines...")
    res = pd.read_csv(docname, error_bad_lines=False, warn_bad_lines=False, encoding="utf8")
    pairs = []
    for r in range(0, res['function_name'].size):
        pair = []
        code = res['function_name'][r]
        target = res['first_sent'][r]
        pair.append(normalizeString(code))
        pair.append(normalizeString(target))
        pairs.append(pair)

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, docname, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, docname, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h).to(device)
        g.ndata['z'] = z 
        g.apply_edges(self.edge_attention, etype='AST')  
        if 'CFG' in g.etypes:
            g.apply_edges(self.edge_attention, etype='CFG')
        if 'DFG' in g.etypes:
            g.apply_edges(self.edge_attention, etype='DFG')
        if 'PDG' in g.etypes:
            g.apply_edges(self.edge_attention, etype='PDG')

        g.update_all(self.message_func, self.reduce_func, etype='AST')
        if 'CFG' in g.etypes:
            g.update_all(self.message_func, self.reduce_func, etype='CFG')
        if 'DFG' in g.etypes:
            g.update_all(self.message_func, self.reduce_func, etype='DFG')
        if 'PDG' in g.etypes:
            g.update_all(self.message_func, self.reduce_func, etype='PDG')
        return g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        self.hidden_dim = hidden_dim
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)
        return h

# 带attention的解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)  

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1) 
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),  
                                 encoder_outputs.unsqueeze(0))  

        output = torch.cat((embedded[0], attn_applied[0]), 1)  
        output = self.attn_combine(output).unsqueeze(0) 
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 准备训练数据
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# 训练模型
def train(g, input_tensor, target_tensor,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion,
          max_length=MAX_LENGTH):

    # 初始化
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs = encoder(g, input_tensor.to(device)).to(device)
    encoder_outputs1 = torch.zeros((max_length - encoder_outputs.shape[0]), encoder.hidden_dim, device=device)
    encoder_outputs = torch.cat((encoder_outputs, encoder_outputs1), 0)

    target_length = target_tensor.size(0)
    loss = 0

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden, best_dst = encoder_outputs.max(0)
    decoder_hidden = decoder_hidden.unsqueeze(0).unsqueeze(1)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# 验证模型
def validate(g, input_tensor, target_tensor,
             encoder, decoder,
             criterion,
             max_length=MAX_LENGTH):

    encoder_outputs = encoder(g, input_tensor.to(device)).to(device)
    encoder_outputs1 = torch.zeros((max_length - encoder_outputs.shape[0]), encoder.hidden_dim, device=device)
    encoder_outputs = torch.cat((encoder_outputs, encoder_outputs1), 0)

    target_length = target_tensor.size(0)
    loss = 0

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden, best_dst = encoder_outputs.max(0)
    decoder_hidden = decoder_hidden.unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])

                decoder_input = target_tensor[di]
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di])

                if decoder_input.item() == EOS_token:
                    break

    return loss.item() / target_length

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

def creat_mini_batch(X, Y, mini_batch_size,seed):
    np.random.seed(seed) #指定随机种子
    m = len(X)
    mini_batches = []

    permutation = list(np.random.permutation(m))
    X_shuffled = []
    Y_shuffled = []
    for per in permutation:
        X_shuffled.append(X[per])
        Y_shuffled.append(Y[per])

    num_minibatches = math.floor(m / mini_batch_size)
    for k in range(0,num_minibatches):
        mini_batch_x = X_shuffled[k*mini_batch_size:(k+1) * mini_batch_size]
        mini_batch_y = Y_shuffled[k*mini_batch_size:(k+1) * mini_batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_x = X_shuffled[mini_batch_size * num_minibatches:]
        mini_batch_y = Y_shuffled[mini_batch_size * num_minibatches:]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches

def trainIters(Gs_train, pairs_train, Gs_valid, pairs_valid, encoder, decoder, n_iters, learning_rate=0.001 ,
                mini_batch_size = 64, early_stop_num = 10):
    start = time.time()
    plot_losses_train = []
    plot_losses_valid = []
    best_loss = None
    early_stop = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    seed = 10
    for iter in range(1, n_iters + 1):
        seed = seed + 1

        mini_batches = creat_mini_batch(Gs_train, pairs_train, mini_batch_size, seed)
        for mini_batch in mini_batches:
            (mini_batch_x, mini_batch_y) = mini_batch
            for smallnum in range(0,len(mini_batch_x)):
                input_tensor = mini_batch_x[smallnum].ndata['feat']
                target_tensor = tensorsFromPair(input_lang1, output_lang1, mini_batch_y[smallnum])[1]
                loss_train = train(mini_batch_x[smallnum], input_tensor, target_tensor, encoder,
                                decoder, encoder_optimizer, decoder_optimizer, criterion)

        for num in range(0, len(Gs_valid)):
            input_tensor = Gs_valid[num].ndata['feat']
            target_tensor = tensorsFromPair(input_lang3, output_lang3, pairs_valid[num])[1]
            loss_valid = validate(Gs_valid[num], input_tensor, target_tensor, encoder, decoder,  criterion)
        if best_loss is None or best_loss > loss_valid:
            best_loss = loss_valid
            early_stop = 0
        else:
            early_stop += 1
        print('%s (%d %d%%)' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100),
                                      'train_loss: ', '%.4f' % loss_train,
                                      '; valid_loss: ', '%.4f' % loss_valid,
                                      '; best_loss: ', '%.4f' % best_loss)
        if early_stop >= early_stop_num:
            print('early stop')
            return

def evaluate(g, encoder, decoder, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = g.ndata['feat']

        encoder_outputs = encoder(g, input_tensor.to(device)).to(device)
        encoder_outputs1 = torch.zeros((max_length - encoder_outputs.shape[0]), encoder.hidden_dim, device=device)

        encoder_outputs = torch.cat((encoder_outputs, encoder_outputs1), 0)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden, best_dst = encoder_outputs.max(0)
        decoder_hidden = decoder_hidden.unsqueeze(0).unsqueeze(1)

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang2.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(Gs, pairs, encoder, decoder, n=10):
    for i in range(n):
        num = random.randint(1, len(Gs))
        input_tensor = Gs[num].ndata['feat']
        g = Gs[num]
        target_tensor = tensorsFromPair(input_lang2, output_lang2, pairs[num])[1]
        print('>', pairs[num][0])
        print('=', pairs[num][1])
        output_words, attentions = evaluate(g, encoder, decoder)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def calculate_bleu(Gs, pairs, encoder, decoder):
    trgs = []
    pred_trgs = []
    smooth_func = SmoothingFunction().method0  # method0 - method7
    meteor = 0

    for num in range(0, len(Gs)):
        src = Gs[num]
        trg = pairs[num][1]

        output_words, attentions = evaluate(src, encoder, decoder)
        output_sentence = ' '.join(output_words)

        meteor_pred_trgs = output_sentence[:-5]
        if len(meteor_pred_trgs) == 0:
            meteor_pred_trgs = ' '
        meteor_trgs = trg

        pred_trg = output_sentence.replace('.', '').split(' ')[:-1]
        if len(pred_trg) == 0:
            pred_trg = [' ']
        pred_trgs.append(pred_trg)
        trg = trg.replace('.', '').split(' ')
        trgs.append(trg)

        tot = 0.0
        for (ref, candi) in zip(meteor_pred_trgs, meteor_trgs):
            tot += round(single_meteor_score(ref, candi), 4)
        meteor +=  tot / len(meteor_pred_trgs)

        rouge = Rouge()
        scores = rouge.get_scores(meteor_pred_trgs, meteor_trgs)

    meteor_all = meteor / len(Gs)

    print('bleu_score: ', corpus_bleu(pred_trgs, trgs, smoothing_function=smooth_func, weights=(1.0,)),
          corpus_bleu(pred_trgs, trgs, smoothing_function=smooth_func, weights=(1.0/2,)*2),
          corpus_bleu(pred_trgs, trgs, smoothing_function=smooth_func, weights=(1.0/3,)*3),
          corpus_bleu(pred_trgs, trgs, smoothing_function=smooth_func, weights=(1.0/4,)*4))
    print('meteor_score: ', meteor_all)
    print('rouge_score:', scores[0]['rouge-1']['r'], scores[0]['rouge-2']['r'], scores[0]['rouge-l']['r'])

    return corpus_bleu(pred_trgs, trgs, smoothing_function=smooth_func, weights=(1.0/4,)*4)

# input_lang1, output_lang1, pairs1 = prepareData('code', 'target', 'data/train.csv', False)
# input_lang2, output_lang2, pairs2 = prepareData('code', 'target', 'data/test.csv', False)
# input_lang3, output_lang3, pairs3 = prepareData('code', 'target', 'data/valid.csv', False)
input_lang1, output_lang1, pairs1 = prepareData('code', 'target', 'data/100.csv', False)
input_lang2, output_lang2, pairs2 = prepareData('code', 'target', 'data/100.csv', False)
input_lang3, output_lang3, pairs3 = prepareData('code', 'target', 'data/100.csv', False)

teacher_forcing_ratio = 0.5
hidden_size = 512

encoder = GAT(in_dim=512, hidden_dim=512, out_dim=512, num_heads=8).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang1.n_words, dropout_p=0.1).to(device)

embed1 = nn.Embedding(4, 512)
# funcnamelist1 = pd.read_csv("data/train.csv",error_bad_lines=False,warn_bad_lines=False,encoding="utf8")
funcnamelist1 = pd.read_csv("data/100.csv",error_bad_lines=False,warn_bad_lines=False,encoding="utf8")
Gs1 = []
Gs1 = createGraph.create_graphlist(funcnamelist1, embed1)
# funcnamelist2 = pd.read_csv("data/test.csv",error_bad_lines=False,warn_bad_lines=False,encoding="utf8")
funcnamelist2 = pd.read_csv("data/100.csv",error_bad_lines=False,warn_bad_lines=False,encoding="utf8")
Gs2 = []
Gs2 = createGraph.create_graphlist(funcnamelist2, embed1)
# funcnamelist3 = pd.read_csv("data/valid.csv",error_bad_lines=False,warn_bad_lines=False,encoding="utf8")
funcnamelist3 = pd.read_csv("data/100.csv",error_bad_lines=False,warn_bad_lines=False,encoding="utf8")
Gs3 = []
Gs3 = createGraph.create_graphlist(funcnamelist3, embed1)

trainIters(Gs1, pairs1, Gs3, pairs3, encoder, attn_decoder1, 100, mini_batch_size = 10, early_stop_num = 3)

# evaluateRandomly(Gs2, pairs2, encoder, attn_decoder1)
calculate_bleu(Gs2, pairs2, encoder, attn_decoder1)

