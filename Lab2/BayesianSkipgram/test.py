import pickle
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import argparse
import logging
import torch
import gensim
import numpy as np


def test(w2i, pairs, encoder, decoder, enable_cuda, priors):
    outputs = []
    for orig_centre, context, n, term, candidates, candidates_rest in pairs:
        if orig_centre not in w2i: 
            centre = term.split('.')[0]
        else:
            centre = orig_centre
        centre_tensor = torch.autograd.Variable(torch.LongTensor([w2i[centre] if centre in w2i else w2i["UNK"]]))
        context = torch.autograd.Variable(torch.LongTensor([[w2i[w] for w in context if w in w2i]]))
        if enable_cuda:
            centre_tensor = centre_tensor.cuda()
            context = context.cuda()

        ranking = Counter()
        for candidate in candidates:
            if not centre in w2i:
                ranking[candidate] = 0
            else:
                candidate_tensor = torch.autograd.Variable(torch.LongTensor([w2i[candidate]]))
                if enable_cuda: candidate_tensor = candidate_tensor.cuda()
                if priors:
                    mu_w, sigma_w = encoder.forward(centre_tensor, context, True)
                    mu_s, sigma_s = encoder.forward(candidate_tensor, context, True)
                    KL = decoder.KL(mu_w, sigma_w, mu_s, sigma_s)
                else:
                    mu_w, sigma_w = encoder.forward(centre_tensor, context, True)
                    ll, KL = decoder.forward(mu_w, sigma_w, candidate_tensor, context, 1)
                ranking[candidate] = - 1 * KL.data[0]

        # Use format specified in LST README
        output = "RANKED\t{} {}".format(term, n)
        for j, (candidate, score) in enumerate(ranking.most_common()):
            output += "\t{} {}".format(candidate, score)
        for candidate in candidates_rest:
            output += "\t{} {}".format(candidate, 0)
        outputs.append(output)

    with open("test.out", 'w') as f:
        f.write("\n".join(outputs))
    os.system("python ../data/lst/lst_gap.py ../data/lst/lst_test.gold test.out out no-mwe")

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
        description='Bayesian Skipgram testing.')
    p.add_argument('--encoder', type=str, default='encoder_epoch_1.pt',
                   help='Path to pickled embeddings.')
    p.add_argument('--decoder', type=str, default='decoder_epoch_1.pt',
                   help='Path to pickled embeddings.')
    p.add_argument('--w2i', type=str, default='w2i.pickle')
    p.add_argument('--test_sentences', type=str, default="../data/lst/lst_test.preprocessed",
                   help='Sentences for LST task.')
    p.add_argument('--test_candidates', type=str, default="../data/lst/lst.gold.candidates",
                   help='Candidates for LST task.')
    p.add_argument('--window', type=int, default=5, help='Symmetric context window.')
    p.add_argument('--priors', action='store_true')

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    words =  pickle.load(open(args.w2i, 'rb'))
    w2i = dict()
    for key, value in words:
        w2i[key] = int(value)

    encoder = torch.load(args.encoder)
    decoder = torch.load(args.decoder)

    embeddings = dict()
    for key, index in w2i.items():
        embeddings[key] = np.array(decoder.affine.weight.data[index, :])
    pickle.dump(embeddings, open("bsg_prior.pickle", 'wb'))

    embeddings = dict()
    for key, index in w2i.items():
        embeddings[key] = np.array(decoder.L.weight.data[index, :])
    pickle.dump(embeddings, open("bsg_posterior.pickle", 'wb'))


    logging.info("Prepared data, starting testing now.")
    pairs = prepare_test(w2i, args.window, args.test_sentences, args.test_candidates)
    test(w2i, pairs, encoder, decoder, encoder.enable_cuda, args.priors)

    