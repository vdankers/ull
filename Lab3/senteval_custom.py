#cell 1

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
#import data 
# data.py is part of Senteval and it is used for loading word2vec style files
import senteval
import tensorflow as tf
import logging
import copy
import torch
import pickle
from collections import defaultdict, Counter
import dill
import dgm4nlp
from Encoder import Encoder

print('imports complete')

#cell 2

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
# for example the path to a dictionary of indices, of hyper parameters
# this dictionary is passed to the batched and the prepare fucntions
params_senteval = {'task_path': '',
                   'usepytorch': False,
                   'kfold': 10,
                   'ckpt_path': '',
                   'tok_path': '',
                   'extractor': None,
                   'tks1': None,
                   'probs': None,
                   'encoder': None,
                   'w2i': None,
                   'method': 'sum'}

# made dictionary a dotdict
params_senteval = dotdict(params_senteval)
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False 
#params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                 'tenacity': 3, 'epoch_size': 2}



def prepare(params, samples):
    """
    In this example we are going to load a tensorflow model, 
    we open a dictionary with the indices of tokens and the computation graph
    """
    params.extractor = pickle.load(open(params.embeddings, 'rb'))
    # # load tokenizer from training
    # params.tks1 = { w : i for w, i in pickle.load(open(params.tok_path, 'rb')) }
    return

def batcher(params, batch):
    """
    At this point batch is a python list containing sentences. Each sentence is a list of tokens (each token a string).
    The code below will take care of converting this to unique ids that EmbedAlign can understand.
    
    This function should return a single vector representation per sentence in the batch.
    In this example we use the average of word embeddings (as predicted by EmbedAlign) as a sentence representation.
    
    In this method you can do mini-batching or you can process sentences 1 at a time (batches of size 1).
    We choose to do it 1 sentence at a time to avoid having to deal with masking. 
    
    This should not be too slow, and it also saves memory.
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    if params.method == "avg" or params.method == "sif":
        embeddings = []
        for sent in batch:
            # Only keep words that are in your embeddings
            sent = [w.strip() for w in sent if w in params.extractor]
            if sent == []: sent = ['.']

            # sentence vector is the mean of word embeddings in context
            weight = params.probs[sent[0]] if params.probs is not None else 1
            sent_vec = weight * copy.deepcopy(params.extractor[sent[0]])
            for word in sent[1:]:
                weight = params.probs[word] if params.probs is not None else 1
                sent_vec += weight * copy.deepcopy(params.extractor[word])
            sent_vec = sent_vec / np.linalg.norm(sent_vec)
            
            # check if there is any NaN in vector (they appear sometimes when there's padding)
            if np.isnan(sent_vec.sum()):
                sent_vec = np.nan_to_num(sent_vec)        
            embeddings.append(sent_vec)

    else:
        embeddings = []
        for sent in batch:
            # Only keep words that are in your embeddings
            sent = ['<s>'] + [w.strip() for w in sent if w in params.w2i] + ['</s>']
            if sent == []: sent = ['.']
            sent_indices = [params.w2i[w] for w in sent]
            sent_vec = params.encoder.eval(torch.LongTensor([sent_indices])).data[0].numpy()
            #sent_vec = sent_vec / np.linalg.norm(sent_vec)      
            embeddings.append(sent_vec)


    if params.method == "sif":
        u, _, _ = np.linalg.svd(np.matrix(embeddings).transpose())
        u = u[0,:]
        u_matrix = u.transpose() * u
        for i, sent_embedding in enumerate(embeddings):
            sent_embedding = np.matrix(sent_embedding).transpose()
            sent_embedding = sent_embedding - u_matrix * sent_embedding
            embeddings[i] = np.squeeze(np.asarray(sent_embedding.transpose().flatten()))
    embeddings = np.vstack(embeddings)
    return embeddings


def prepare_weighting(corpus, a=0.001):
    probs = Counter()
    with open(corpus, 'r') as f:
        for line in f:
            line = line.split()
            for w in line: probs[w] += 1

    total = sum(probs.values())

    for p in probs:
        pw = (probs[p] / total)
        probs[p] = a / (a + pw)
    return probs


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # define paths, path to senteval data
    # note senteval adds downstream into the path
    params_senteval.task_path = './SentEval/data/'
    params_senteval.method = sys.argv[1]
    params_senteval.embeddings = sys.argv[2]

    if len(sys.argv) > 2:
        if params_senteval.method == "rnn":
            params_senteval.encoder = torch.load(sys.argv[3]).cpu()
            params_senteval.encoder.enable_cuda = False
            w2i = pickle.load(open(sys.argv[4], 'rb'))
            params_senteval.w2i = { w: v for w, v in w2i}
            #exit()
        else:
            if len(sys.argv) > 3:
                params_senteval.probs = prepare_weighting(sys.argv[3], float(sys.argv[4]))
            else:
                print("No a-value specified. Defaulting to unweighted average.")

    # we use 10 fold cross validation
    params_senteval.kfold = 10
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']
    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    logging.info(str(results))