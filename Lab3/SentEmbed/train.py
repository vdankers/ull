import argparse
import logging
import pickle
import torch
import random
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
from test import greedy

# Beginning and end of sentence tokens
BOS = "<s>"
EOS = "</s>"


def clean(sequence):
    """Remove BOS en EOS tokens, they should not be taken into account in BLEU.

    Args:
        sequence (list of str): sequence containing BOS and/or EOS tags

    Returns:
        sequence with BOS and EOS tags removed
    """
    if BOS in sequence: sequence.remove(BOS)
    if EOS in sequence: sequence.remove(EOS)
    return sequence


def validate(corpus, valid, max_length, enable_cuda, epoch):
    """Calculate the BLEU scores for the validation data.

    Args:
        corpus (Corpus): containing w2i and i2w dictionaries
        valid (two lists): english and english sentences
        max_length (int): maximum length of generated translation
        enable_cuda (bool): whether GPU is available
        epoch (int): which epoch we are at now

    Returns:
        BLEU score (int)
    """
    scores = []
    chencherry = SmoothingFunction()
    shuffle(valid)
    for english in valid[:5000]:
        indices = corpus.to_indices(english)
        decoding = greedy(
            encoder, decoder, indices, corpus.dict.word2index,
            corpus.dict.index2word, max_length, enable_cuda
        )

        scores.append(sentence_bleu([english], decoding,
                                     smoothing_function=chencherry.method1))

    score = sum(scores) / len(scores)
    logging.info("Greedy average BLEU score: {}".format(score))
    return score


def train(corpus, valid, encoder, decoder, lr, epochs, batch_size, enable_cuda,
          ratio, max_length):
    """Calculate the BLEU scores for the validation data.

    Args:
        corpus (Corpus): containing w2i and i2w dictionaries
        valid (two lists): english and english sentences
        encoder (Encoder): custom encoder object
        decoder (Decoder): custom decoder object
        lr (float): learning rate
        epochs (int): number of learning iterations to train the model
        batch_size (int): the batch size
        enable_cuda (bool): whether GPU is available
        ratio (float): ratio for teacher forcing
        max_length (int): maximum length of generated translation

    Returns:
        losses (list of floats)
        bleus (list of floats)
    """
    criterion = nn.NLLLoss()
    params_enc = list(filter(lambda p: p.requires_grad, encoder.parameters()))
    params_dec = list(filter(lambda p: p.requires_grad, decoder.parameters()))
    optimizer = torch.optim.Adam(params_enc + params_dec, lr=lr)
    losses = []
    bleus = []
    for i in range(epochs):
        epoch_loss = 0

        for english in corpus.batches:
            optimizer.zero_grad()
            use_teacher_forcing = True if random() < ratio else False

            # First run the encoder, it encodes the English sentence
            h_dec = encoder(english)
            c = h_dec

            # Now go through the decoder step by step and use teacher forcing
            # for a ratio of the batches
            next_token = english[:, 0]
            loss = 0
            for j in range(1, english.shape[1]):
                vocab_probs, h_dec, c = decoder(next_token, h_dec, c)
                loss += criterion(vocab_probs, english[:, j])
                if use_teacher_forcing:
                    next_token = english[:, j]
                else:
                    _, next_token = torch.topk(vocab_probs, 1)
                    next_token = next_token[:, 0]

            # Now update the weights
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / english.shape[1]

        # Dropout annealing
        encoder.dropout_rate = max([0, 1 - i / 5]) * encoder.dropout_rate_0
        decoder.dropout_rate = max([0, 1 - i / 5]) * decoder.dropout_rate_0

        epoch_loss = epoch_loss / len(corpus.batches)
        logging.info("Loss per token: {}".format(epoch_loss))
        losses.append(epoch_loss)
    bleus.append(validate(corpus, valid, max_length, enable_cuda, i))
    return losses, bleus


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--english',       type=str,   default='../data/europarl/training.en')
    p.add_argument('--enc_type',      type=str,   default='avg')
    p.add_argument('--embed',         type=str,   default='sgns.pickle')
    p.add_argument('--lr',            type=float, default=0.0005)
    p.add_argument('--tf_ratio',      type=float, default=0.75)
    p.add_argument('--batch_size',    type=int,   default=128)
    p.add_argument('--epochs',        type=int,   default=10)
    p.add_argument('--dim',           type=int,   default=100)
    p.add_argument('--min_count',     type=int,   default=1)
    p.add_argument('--max_length',    type=int,   default=50)
    p.add_argument('--lower',         action='store_true')
    p.add_argument('--enable_cuda',   action='store_true')

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
    embed_dict = pickle.load(open(args.embed, 'rb'))

    # Prepare corpus, encoder and decoder
    corpus = Corpus(args.english, args.batch_size, args.min_count, args.lower,
                    args.enable_cuda, embed_dict)

    dim = len(embed_dict["and"])
    embeddings = torch.Tensor(len(embed_dict), dim)
    for word in embed_dict:
        try:
            embeddings[corpus.dict.word2index[word]] = torch.from_numpy(embed_dict[word])
        except:
            print(word)

    print(len(list(embed_dict.keys())), len(corpus.dict.word2index))

    encoder = Encoder(embeddings, dim, corpus.vocab_size, corpus.max_pos, args.enc_type, enable_cuda)
    eos = corpus.dict.word2index["</s>"]
    decoder = Decoder(embeddings, dim, corpus.vocab_size, eos, corpus.longest_english)
    if enable_cuda:
        encoder.cuda()
        decoder.cuda()
    valid = corpus.load_data(args.english)

    # Train
    losses, bleus = train(corpus, valid, encoder, decoder, args.lr, args.epochs,
                          args.batch_size, enable_cuda, args.tf_ratio,
                          args.max_length)

    torch.save(encoder, "encoder_{}.pt".format(args.embed.split(".")[0]))
    torch.save(decoder, "decoder_{}.pt".format(args.embed.split(".")[0]))
    word2index = list(corpus.dict.word2index.items())
    index2word = list(corpus.dict.index2word.items())
    pickle.dump(word2index, open("w2i.pickle", 'wb'))