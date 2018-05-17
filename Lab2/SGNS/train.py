import argparse
import logging
import pickle
import torch
from tqdm import tqdm
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
from torch import optim
from collections import defaultdict
from random import shuffle
from data import Corpus
from SGNS import SGNS
import copy
from test import prepare_test, test
from matplotlib import pyplot as plt
from test import to_dict
from sklearn.decomposition import PCA


def train(batches, sgns, epochs, lr, batch_size, enable_cuda, test_pairs, embed_file, gold, words):
    criterion = nn.NLLLoss()
    losses = []
    optimizer = optim.Adam(sgns.parameters(), lr=lr) 
    for i in range(epochs):
        embeddings, _ = to_dict(copy.deepcopy(sgns), words, embed_file)

        test(embeddings, test_pairs, False, gold)

        all_loss = 0
        logging.info("Epoch {}".format(i+1))
        for (neighbour, centre, neg_samples) in batches:
            optimizer.zero_grad()
            loss = sgns.forward(centre, neighbour, neg_samples)
            loss.backward()
            optimizer.step()
            all_loss += loss.data[0] 
        losses.append(all_loss / len(batches) / batch_size)
        logging.info("Average loss per training sample: {}".format(all_loss / len(batches) / batch_size))
        pickle.dump(corpus.words, open("words.pickle", 'wb'))
        torch.save(sgns, "epoch_{}.pt".format(i))

    return sgns, losses

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description='Skipgram Negative Sampling.')
    p.add_argument('--corpus', type=str, default='data/train.txt',
                   help='path to word-association pairs for training.')
    p.add_argument('--lr', type=float, default=0.01, help='learning rate')
    p.add_argument('--batch_size', type=int, default=10, help='batch size')
    p.add_argument('--enable_cuda', action='store_true', help='use CUDA')
    p.add_argument('--save', type=str, help='path for saving model')
    p.add_argument('--embed_file', default='SGNS.pickle', type=str, help='file to save embeddings')
    p.add_argument('--epochs', type=int, default=10, help='#epochs')
    p.add_argument('--window', default=5, type=int)
    p.add_argument('--dim', default=300, type=int)
    p.add_argument('--nr_sents', default=-1, type=int)
    p.add_argument('--neg_samples', default=5, type=int)
    p.add_argument('--min_count', default=1, type=int)
    p.add_argument('--candidates', default='../data/lst/lst.gold.candidates')
    p.add_argument('--valid', default='../data/lst/lst_valid.preprocessed')
    p.add_argument('--gold', default='../data/lst/lst_valid.gold')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Check whether GPU is present
    if args.enable_cuda and torch.cuda.is_available():
        enable_cuda = True
        logging.info("CUDA is enabled")
        torch.cuda.set_device(0)
    else:
        enable_cuda = False
        logging.info("CUDA is disabled")

    # Prepare corpus + dictionaries, create training batches
    corpus = Corpus(args.corpus, args.window, args.min_count, args.batch_size, args.nr_sents, args.neg_samples, enable_cuda)
    logging.info("Loaded data.")

    # Initialize model and cuda if necessary
    sgns = SGNS(corpus.vocab_size, args.dim, enable_cuda)
    # torch.save(lm, "models/start_0.pt")
    if enable_cuda:
        sgns.cuda()

    # Train
    logging.info("Training will start shortly.")
    pairs = prepare_test(corpus.dictionary.word2index, args.window, args.valid, args.candidates)
    sgns, losses = train(corpus.batches, sgns, args.epochs, args.lr, args.batch_size, enable_cuda, pairs, args.embed_file, args.gold, corpus.words)

    # Plot vectors using TSNE technique
    plt.figure(figsize=(40, 30))
    pca = PCA(n_components=2)

    embeddings, matrix = to_dict(sgns, args.embed_file)
    pca_result = pca.fit_transform(matrix)
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    for i, word in enumerate(corpus.words):
        plt.annotate(word, xy=(pca_result[i, 0], pca_result[i, 1]), xytext=(0, 0), textcoords='offset points')
    plt.savefig("tsne.png")

    plt.figure(figsize=(15, 10))
    plt.scatter([i for i in range(len(losses))], losses, alpha=0.3)
    plt.savefig("loss.png")
