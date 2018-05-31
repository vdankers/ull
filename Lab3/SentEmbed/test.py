import argparse
import logging
import pickle
import torch
import os
import torch.nn as nn
from matplotlib import pyplot as plt

from tqdm import tqdm
from torch import optim
from collections import defaultdict, Counter
from random import shuffle, random
from data import Corpus
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import plotly.plotly as py 
import plotly.graph_objs as go
import plotly.offline as offline
import plotly
plotly.tools.set_credentials_file(username='vdankers', api_key='iqYAxJNr16lmrFjgys4l')

from Encoder import Encoder
from Decoder import Decoder

# Beginning and end of sentence tokens
BOS = "<s>"
EOS = "</s>"


def greedy(encoder, decoder, english, w2i, i2w, max_length,
           enable_cuda):
    """Generate a translation one by one, by greedily selecting every new
    new word.

    Args:
        encoder: custom Encoder object
        decoder: custom Decoder object
        english: list of English words
        positions: list of word positions
        w2i (dict): mapping words to indices
        i2w (dict): mapping indices to words
        max_length (int): maximum generation length
        enable_cuda (bool): whether to enable CUDA

    Returns:
        translation: list of words
    """

    # Initialise the variables and the encoding
    english, next_token = prepare_variables(
        english, w2i, enable_cuda
    )
    translation = []
    h_dec = encoder.eval(english)
    c = h_dec

    # Sample words one by one
    for j in range(1, max_length + 1):
        vocab_probs, h_dec, c = decoder.eval(next_token, h_dec, c)
        _, next_token = torch.topk(vocab_probs, 1)
        next_token = next_token[:, 0]
        translation.append(i2w[next_token.item()])

        # If the </s> token is found, end generating
        if translation[-1] == EOS:
            break

    return translation


def clean(sequence):
    # Remove BOS en EOS tokens because they should not be taken into account in BLEU
    if BOS in sequence: sequence.remove(BOS)
    if EOS in sequence: sequence.remove(EOS)
    return sequence

def prepare_variables(english, w2i, enable_cuda):
    """Turn words and positions into Variables containing Longtensors.

    Args:
        english (list): list of english words
        w2i (dict): mapping words to indices
        enable_cuda (bool): whether cuda is available

    Returns;
        english: now a Variable containing a Longtensor
        positions: now a Variable containing a Longtensor
        next_token: Variable with the first french token "<s>"
    """
    english = torch.autograd.Variable(torch.LongTensor([english]))
    next_token = torch.autograd.Variable(torch.LongTensor([w2i[BOS]]))

    if enable_cuda:
        english = english.cuda()
        next_token = next_token.cuda()
    return english, next_token


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--english', type=str,   default='test/test_2017_flickr.en')
    p.add_argument('--french',  type=str,   default='test/test_2017_flickr.fr')
    p.add_argument('--encoder', type=str,   default='encoder_type=gru.pt')
    p.add_argument('--decoder', type=str,   default='decoder_type=gru.pt')
    p.add_argument('--corpus',  type=str,   default='corpus.pickle')
    p.add_argument('--max_length',    	type=int,   default=74)
    p.add_argument('--enable_cuda',   	action='store_true')
    p.add_argument('--transformer', 	action='store_true')

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Check whether GPU is present
    if args.enable_cuda and torch.cuda.is_available():
        enable_cuda = True
        torch.cuda.set_device(1)
        logging.info("CUDA is enabled")
    else:
        enable_cuda = False
        logging.info("CUDA is disabled")

    encoder = torch.load(args.encoder)
    decoder = torch.load(args.decoder)
    corpus = pickle.load(open(args.corpus, 'rb'))
    corpus.dict_e.word2index = { k: v for k, v in corpus.dict_e.word2index }
    corpus.dict_e.index2word = { k: v for k, v in corpus.dict_e.index2word }
    corpus.dict_f.word2index = { k: v for k, v in corpus.dict_f.word2index }
    corpus.dict_f.index2word = { k: v for k, v in corpus.dict_f.index2word }
    test_pairs = corpus.load_data(args.english, args.french)
    test(corpus, test_pairs, args.max_length, enable_cuda, 0, args.transformer)