import argparse
import logging
import pickle
import torch
import os
from tqdm import tqdm
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
from torch import optim
from collections import defaultdict, Counter
from random import shuffle
from data import Corpus
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from Encoder import Encoder
from Decoder import Decoder


def validate(corpus, pairs, encoder, decoder, i, enable_cuda):
    w2i = corpus.dictionary.word2index
    i2w = corpus.dictionary.index2word
    outputs = []
    for orig_centre, context, n, term, candidates, candidates_rest in pairs:
        if orig_centre not in w2i: 
            centre = term.split('.')[0]
        else:
            centre = orig_centre
        centre = torch.autograd.Variable(torch.LongTensor([w2i[centre]]))
        context = torch.autograd.Variable(torch.LongTensor([[w2i[w] for w in context if w in w2i]]))
        if enable_cuda:
            centre = centre.cuda()
            context = context.cuda()

        ranking = Counter()
        for candidate in candidates:
            if not orig_centre in w2i and not term.split('.')[0] in w2i:
                ranking[candidate] = 0
            else:
                candidate_tensor = torch.autograd.Variable(torch.LongTensor([w2i[candidate]]))
                if enable_cuda: candidate_tensor = candidate_tensor.cuda()
                mu_w, sigma_w = encoder.forward(centre, context, True)
                mu_s, sigma_s = encoder.forward(candidate_tensor, context, True)
                KL = decoder.KL(mu_w, sigma_w, mu_s, sigma_s)
                ranking[candidate] = - 1 * KL.data[0]

        # Use format specified in LST README
        output = "RANKED\t{} {}".format(term, n)
        for j, (candidate, score) in enumerate(ranking.most_common()):
            output += "\t{} {}".format(candidate, score)
        for candidate in candidates_rest:
            output += "\t{} {}".format(candidate, 0)
        outputs.append(output)

    with open("epoch_{}.out".format(i + 1), 'w') as f:
        f.write("\n".join(outputs))
    print(i)
    os.system("python ../data/lst/lst_gap.py ../data/lst/lst_valid.gold epoch_{}.out out no-mwe".format(i + 1))


def train(corpus, encoder, decoder, epochs, lr, batch_size, enable_cuda, test_pairs):
    criterion = nn.NLLLoss()
    losses = []
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    n = len(corpus.batches)
    for i in range(epochs):
        validate(corpus, test_pairs, encoder, decoder, i, enable_cuda)
        all_loss = 0
        all_ll = 0
        all_KL = 0
        logging.info("Epoch {}".format(i+1))
        for (centre, context) in corpus.batches:
            optimizer.zero_grad()
            mu, sigma = encoder.forward(centre, context)
            ll, KL = decoder.forward(mu, sigma, centre, context)
            loss = -1 * (ll - KL)
            all_ll += ll.data[0]
            all_KL += KL.data[0]
            loss.backward()
            optimizer.step()
            all_loss += loss.data[0]

        pickle.dump(list(corpus.dictionary.word2index.items()), open("w2i.pickle".format(i), 'wb'))
        torch.save(encoder, "encoder_epoch_{}.pt".format(i))
        torch.save(decoder, "decoder_epoch_{}.pt".format(i))

        losses.append(all_loss / n / batch_size)
        logging.info("Average loss per training sample: {}".format(all_loss / n / batch_size))
        logging.info("KL {}, LL {}".format(all_KL / n / batch_size, all_ll / n / batch_size))
    return losses


def prepare_test(w2i, window, sentences_path="../data/lst/lst_test.preprocessed",
                 cand_path="../data/lst/lst.gold.candidates"):
    test_pairs = []
    candidates = dict()
    missing_candidates = dict()
    with open(cand_path, 'r') as f:
        for line in f:
            term, term_candidates = tuple(line.split("::"))
            term_candidates = term_candidates.strip().split(";")
            candidates[term] = [ c for c in term_candidates if c in w2i]
            missing_candidates[term] = [ c for c in term_candidates if c not in w2i]

    with open(sentences_path, 'r') as f:
        for line in f:
            # Read in data line by line
            term, number, pos, sentence = tuple(line.split("\t"))
            pos = int(pos)
            sentence = sentence.split()

            # Extract the context of the term and pad with <s> or </s>
            pre = sentence[max(pos - window, 0):pos]
            post = sentence[pos+1:min(pos + window + 1, len(sentence))]
            context = pre + post
            centre = sentence[pos]
            test_pairs.append((centre, context, int(number), term, candidates[term], missing_candidates[term]))
    return test_pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description='Skipgram Negative Sampling.')
    p.add_argument('--corpus', type=str, default='data/train.txt',
                   help='path to word-association pairs for training.')
    p.add_argument('--lr', type=float, default=0.01, help='learning rate')
    p.add_argument('--batch_size', type=int, default=10, help='batch size')
    p.add_argument('--enable_cuda', action='store_true', help='use CUDA')
    p.add_argument('--save', type=str, help='path for saving model')
    p.add_argument('--epochs', type=int, default=10, help='#epochs')
    p.add_argument('--window', default=5, type=int)
    p.add_argument('--dim', default=300, type=int)
    p.add_argument('--nr_sents', default=-1, type=int)
    p.add_argument('--neg_samples', default=5, type=int)
    p.add_argument('--dim2', default=300, type=int)
    p.add_argument('--candidates', default='../data/lst/lst.gold.candidates')
    p.add_argument('--valid', default='../data/lst/lst_valid.preprocessed')
    p.add_argument('--gold', default='../data/lst/lst_valid.gold')

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Check whether GPU is present
    if args.enable_cuda and torch.cuda.is_available():
        enable_cuda = True
        #torch.cuda.set_device(0)
        logging.info("CUDA is enabled")
    else:
        enable_cuda = False
        logging.info("CUDA is disabled")

    # Prepare corpus + dictionaries, create training batches
    corpus = Corpus(args.corpus, args.window, args.batch_size, args.nr_sents, args.neg_samples, enable_cuda)
    test_pairs = prepare_test(corpus.dictionary.word2index, args.window, args.valid, args.candidates)
    logging.info("Loaded data.")

    # Initialize model and cuda if necessary
    encoder = Encoder(corpus.vocab_size, args.dim, args.dim2, args.window, enable_cuda)
    decoder = Decoder(corpus.vocab_size, args.dim, args.dim2, args.window, args.batch_size, enable_cuda)
    if enable_cuda:
        encoder.cuda()
        decoder.cuda()

    # Train
    logging.info("Training will start shortly.")
    losses = train(corpus, encoder, decoder, args.epochs, args.lr, args.batch_size, enable_cuda, test_pairs)

    # # Plot vectors using TSNE technique
    # plt.figure(figsize=(40, 30))
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(sgns.in_embeddings.cpu().weight.data.numpy())
    # plt.scatter(pca_result[:, 0], pca_result[:, 1])
    # for i, word in enumerate(corpus.words):
    #     plt.annotate(word, xy=(pca_result[i, 0], pca_result[i, 1]), xytext=(0, 0), textcoords='offset points')
    # plt.savefig("tsne.png")

    plt.figure(figsize=(15, 10))
    plt.scatter([i for i in range(len(losses))], losses, alpha=0.3)
    plt.savefig("loss.png")

