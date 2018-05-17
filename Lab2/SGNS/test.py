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


def to_dict(sgns, words, both=False):
    in_matrix = sgns.in_embeddings.cpu().weight.data.numpy()
    in_dict = { word : in_matrix[i, :] for i, word in enumerate(words) }
    if not both:
        return in_dict
    else:
        out_matrix = sgns.out_embeddings.cpu().weight.data.numpy()
        out_dict = { word : out_matrix[i, :] for i, word in enumerate(words) }
        return { key : out_dict[key] + in_dict[key] for key in words }


def test(embeddings, pairs, multiply, gold="../data/lst/lst_test.gold"):

    outputs = []
    for centre, context, n, term, candidates, missing_candidates in pairs:
        if centre not in embeddings: 
            centre = term.split('.')[0]
        context_vector = np.zeros((len(embeddings["work"],)))
        for word in context:
            if word not in embeddings: continue # emb = embeddings["UNK"]
            emb = embeddings[word]
            if multiply:
                context_vector *= np.array(emb)
            else:
                context_vector += np.array(emb)

        ranking = Counter()
        for candidate in candidates:
            if not centre in embeddings:
                ranking[candidate] = 0
            else:    
                if multiply:
                    centre_context = context_vector * np.array(embeddings[centre])
                    candidate_context = context_vector * np.array(embeddings[candidate])
                else:
                    centre_context = context_vector + np.array(embeddings[centre])
                    candidate_context = context_vector + np.array(embeddings[candidate])

                ranking[candidate] = 1 - cosine(centre_context, candidate_context)
        
        # Use format specified in LST README
        output = "RANKED\t{} {}".format(term, n)
        for j, (candidate, score) in enumerate(ranking.most_common()):
            output += "\t{} {}".format(candidate, score)
        for candidate in missing_candidates:
            output += "\t{} {}".format(candidate, 0)
        outputs.append(output)

    # Compute GAP
    with open("sgns.out", 'w') as f:
        f.write("\n".join(outputs))
    os.system("python ../data/lst/lst_gap.py {} sgns.out out no-mwe".format(gold))

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
    p.add_argument('--embed_file', type=str, default='SGNS.pickle',
                   help='Path to pickled embeddings.')
    p.add_argument('--test_sentences', type=str, default="../data/lst/lst_test.preprocessed",
                   help='Sentences for LST task.')
    p.add_argument('--test_candidates', type=str, default="../data/lst/lst.gold.candidates",
                   help='Candidates for LST task.')
    p.add_argument('--window', type=int, default=5, help="Symmetric context window.")
    p.add_argument('--model', action='store_true', help="Whether an entire model is given or just embeddings.")
    p.add_argument('--multiply', action='store_true', help="If flagged, multiplies vectors instead of adding them.")
    p.add_argument('--both', action='store_true', help="If flagged, uses the output layer of SGNS as embeddings.")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.model:
        sgns = torch.load(args.embed_file)
        words = pickle.load(open("words.pickle", 'rb'))
        embeddings = to_dict(sgns, words, args.both)
    else:
        embeddings = pickle.load(open(args.embed_file, 'rb'))

    logging.info("Prepared data, starting testing now.")
    pairs = prepare_test(embeddings, args.window, args.test_sentences, args.test_candidates)
    test(embeddings, pairs, args.multiply)

    