import pickle
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from make_aer_pred import make_pred
import aer
import argparse
import logging
import torch
import gensim
import numpy as np

def get_aer(w2i_e, w2i_f, decoder, encoder):
    # aer
    make_pred('./../data/wa/test.en', './../data/wa/test.fr',
              './alignment.pred', w2i_e, w2i_f, decoder, encoder)
    AER = aer.test('./../data/wa/test.naacl', './alignment.pred')   
    logging.info('alignment error rate: {}'.format(AER))


def test(w2i, w2i_f, pairs, encoder, decoder, enable_cuda):
    get_aer(w2i, w2i_f, decoder, encoder)

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

    with open("embedalign.out", 'w') as f:
        f.write("\n".join(outputs))
    os.system("python ../data/lst/lst_gap.py ../data/lst/lst_test.gold embedalign.out out no-mwe")

def prepare_test(w2i, sentences_path="../data/lst/lst_test.preprocessed",
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
            pre = sentence[:pos]
            post = sentence[pos+1:]
            context = (pre, post)
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
    p.add_argument('--w2i_e', type=str, default='w2i_e.pickle')
    p.add_argument('--w2i_f', type=str, default='w2i_f.pickle')
    p.add_argument('--test_sentences', type=str, default="../data/lst/lst_test.preprocessed",
                   help='Sentences for LST task.')
    p.add_argument('--test_candidates', type=str, default="../data/lst/lst.gold.candidates",
                   help='Candidates for LST task.')
    p.add_argument('--window', type=int, default=5, help='Symmetric context window.')

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    words =  pickle.load(open(args.w2i_e, 'rb'))
    w2i_e = { key : int(value) for key, value in words }
    words =  pickle.load(open(args.w2i_f, 'rb'))
    w2i_f = { key : int(value) for key, value in words }

    encoder = torch.load(args.encoder)
    decoder = torch.load(args.decoder)


    embeddings = dict()
    for key, index in w2i_e.items():
        embeddings[key] = np.array(decoder.affine1.weight.data[index, :])
    pickle.dump(embeddings, open("ea.pickle", 'wb'))

    logging.info("Prepared data, starting testing now.")
    pairs = prepare_test(w2i_e, args.test_sentences, args.test_candidates)
    test(w2i_e, w2i_f, pairs, encoder, decoder, encoder.enable_cuda)

    