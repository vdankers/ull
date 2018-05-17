import argparse
import logging
import pickle
import torch
import os
import aer
from make_aer_pred import make_pred
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

def get_aer(corpus, decoder, encoder):
    # aer
    make_pred('./../data/wa/test.en', './../data/wa/test.fr',
              './alignment.pred', corpus.dictionary_e.word2index,
              corpus.dictionary_f.word2index, decoder, encoder)
    AER = aer.test('./../data/wa/test.naacl', './alignment.pred')   
    logging.info('alignment error rate: {}'.format(AER))

def validate(corpus, pairs, encoder, decoder, i, enable_cuda):
    w2i = corpus.dictionary_e.word2index
    i2w = corpus.dictionary_e.index2word
    outputs = []
    for orig_centre, (pre, post), n, term, candidates, candidates_rest in pairs:
        if orig_centre not in w2i: 
            centre = term.split('.')[0]
        else:
            centre = orig_centre
        original_tensor = torch.autograd.Variable(torch.LongTensor([[w2i[w] for w in pre + [centre] + post if w in w2i]]))
        if enable_cuda: original_tensor = original_tensor.cuda()

        ranking = Counter()
        for candidate in candidates:
            if not centre in w2i:
                ranking[candidate] = 0
            else:
                candidate_tensor = torch.autograd.Variable(torch.LongTensor([[w2i[w] for w in pre + [candidate] + post if w in w2i]]))
                if enable_cuda: candidate_tensor = candidate_tensor.cuda()
                mu_o, sigma_o = encoder.forward(original_tensor)
                mu_c, sigma_c = encoder.forward(candidate_tensor)
                KL = decoder.KL(mu_c, sigma_c, mu_o, sigma_o)
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
    os.system("python ../data/lst/lst_gap.py ../data/lst/lst_valid.gold epoch_{}.out out no-mwe".format(i + 1))

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
            context = (pre, post)
            centre = sentence[pos]
            test_pairs.append((centre, context, int(number), term, candidates[term], missing_candidates[term]))
    return test_pairs


def train(corpus, encoder, decoder, epochs, lr, batch_size, enable_cuda, test_pairs, do_validation=True):
    losses = []
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    n = len(corpus.batches)
    get_aer(corpus, decoder, encoder)
    for i in range(epochs):
        if do_validation:
            validate(corpus, test_pairs, encoder, decoder, i, enable_cuda)
        all_loss = 0
        all_ll = 0
        all_KL = 0
        logging.info("Epoch {}".format(i+1))
        for (english, french) in corpus.batches:
            optimizer.zero_grad()
            mu, sigma = encoder.forward(english)
            ll, KL = decoder.forward(mu, sigma, english, french)
            loss = -1 * (ll - KL)
            all_ll += ll.data[0]
            all_KL += KL.data[0]
            loss.backward()
            optimizer.step()
            all_loss += loss.data[0]     
        get_aer(corpus, decoder, encoder)
        torch.save(encoder, "encoder_epoch_{}.pt".format(i))
        torch.save(decoder, "decoder_epoch_{}.pt".format(i))
        pickle.dump(list(corpus.dictionary_e.word2index.items()), open("w2i_e.pickle", 'wb'))
        pickle.dump(list(corpus.dictionary_f.word2index.items()), open("w2i_f.pickle", 'wb'))

        losses.append(all_loss / n / batch_size)
        logging.info("Average loss per training sample: {}".format(all_loss / n / batch_size))
        logging.info("KL {}, LL {}".format(all_KL / n / batch_size, all_ll / n / batch_size))
    return losses

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description='Skipgram Negative Sampling.')
    p.add_argument('--english', type=str, default='../data/hansards/training.en',
                   help='path to Fnglish data.')
    p.add_argument('--french', type=str, default='../data/hansards/training.fr',
                   help='path to French data.')
    p.add_argument('--lr', type=float, default=0.01, help='learning rate')
    p.add_argument('--batch_size', type=int, default=10, help='batch size')
    p.add_argument('--enable_cuda', action='store_true', help='use CUDA')
    p.add_argument('--save', type=str, help='path for saving model')
    p.add_argument('--epochs', type=int, default=10, help='#epochs')
    p.add_argument('--window', default=5, type=int)
    p.add_argument('--dim', default=50, type=int)
    p.add_argument('--nr_sents', default=-1, type=int)
    p.add_argument('--unique_words', default=10000, type=int)
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
    corpus = Corpus(args.english, args.french, args.batch_size, args.nr_sents, args.unique_words, args.enable_cuda)
    test_pairs = prepare_test(corpus.dictionary_e.word2index, args.window, args.valid, args.candidates)

    # logging.info("Loaded data.")

    # Initialize model and cuda if necessary
    encoder = Encoder(corpus.vocab_size_e, args.dim, enable_cuda)
    decoder = Decoder(corpus.vocab_size_e, corpus.vocab_size_f, args.dim, args.batch_size, enable_cuda)
    if enable_cuda:
        encoder.cuda()
        decoder.cuda()

    # Train
    logging.info("Training will start shortly.")
    losses = train(corpus, encoder, decoder, args.epochs, args.lr, args.batch_size, enable_cuda,test_pairs, True)

    #plt.figure(figsize=(15, 10))
    #plt.scatter([i for i in range(len(losses))], losses, alpha=0.3)
    #plt.savefig("loss.png")
    
    