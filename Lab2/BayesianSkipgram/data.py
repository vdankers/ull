"""Prepare a corpus for definition modelling.

Tasks performed:
    - Generate a vocab dictionary that maps words to indices and vice versa.
    - Preprocess text by adding start and end tags.
    - Read text files and save pairs word - association.
    - Convert pairs to numbers in preparation of training a language model.
"""

import nltk
import sys
import logging
import random
import torch
import pickle
import os
import numpy as np
from tqdm import tqdm
from random import shuffle
from torch.autograd import Variable
from collections import defaultdict, Counter


class Dictionary(object):
    """Object that creates and keeps word2index and index2word dicts."""

    def __init__(self):
        """Initialize word - index mappings."""
        self.word2index = defaultdict(lambda: len(self.word2index))
        self.index2word = dict()
        self.counts = Counter()

    def add_word(self, word):
        """Add one new word to your dicitonary.

        Args:
            word (str): word to add to dictionary
        """
        index = self.word2index[word]
        self.index2word[index] = word
        self.counts[word] += 1
        return index

    def add_text(self, text):
        """Add a list of words to your dictionary.

        Args:
            text (list of strings): text to add to dictionary
        """
        for word in text:
            self.add_word(word)

    def prepare_negative_sampling_table(self):
        """Create a table from which one can randomly draw negative samples.
        Every word is represented according to an adapted frequency count."""
        table = []
        self.words = [self.index2word[i] for i in range(len(self.index2word))]
        adapted_counts = np.array(
            list([float(self.counts[w]) ** (3/4) for w in self.words])
        )
        # print(adapted_counts)
        normalizer = sum(adapted_counts)
        for i, number in enumerate(adapted_counts):
            p = number / normalizer
            p = p * 1000000
            for j in range(int(p)):
                table.append(i)
        logging.info("Prepared negative samples.")
        self.table = table

    def sample(self, pos_index, amount):
        """ Draw the desired amount of negative samples that are not the given index.

        Args:
            pos_index (int): index of the positive word, do not return this as negative sample
            amount (int): number of negative samples to return 

        Returns:
            list of negative samples
        """
        samples = []
        for i in range(amount):
            neg_index = pos_index[0]
            while neg_index in pos_index:
                neg_index = self.table[random.randint(0, len(self.table)-1)]
            samples.append(neg_index)
        return samples

    def to_unk(self):
        """From now on your dictionaries default to UNK for unknown words."""
        unk = self.add_word("UNK")
        self.word2index = defaultdict(lambda: unk, self.word2index)


class Corpus(object):
    """Collects words and corresponding associations, preprocesses them."""

    def __init__(self, path, window, batch_size, nr_docs, neg_samples, enable_cuda=False):
        """Initialize pairs of words and associations.

        Args:
            path (str): file path to read data from
            window (int): window size for corpus pairs
            batch_size (int): int indicating the desired size for batches
            nr_docs (int): how many sentences should be used from the corpus
            neg_samples (int): number of negative samples selected per positive sample
            enable_cuda (bool): whether to cuda the batches
        """
        self.window = window
        self.batch_size = batch_size
        self.dictionary = Dictionary()
        self.lines = []
        self.neg_samples = neg_samples
        self.enable_cuda = enable_cuda

        # Read in the corpus
        with open(path, 'r') as f:
            for line in f:
                line = self.prepare(line)
                self.dictionary.add_text(line)
                self.lines.append(line)
                if len(self.lines) == nr_docs and nr_docs != -1:
                    break

        most_common = set([x[0] for x in self.dictionary.counts.most_common(280000)])

        # Redo, but remove words that occur less than five times
        dictionary_norare = Dictionary()
        for i, line in enumerate(self.lines):
            new_line = self.remove_uncommon_from_list(most_common, line)
            dictionary_norare.add_text(new_line)
            self.lines[i] = line
        if nr_docs != -1: self.lines = self.lines[:nr_docs]
        self.dictionary = dictionary_norare
        self.dictionary.to_unk()
        self.vocab_size = len(self.dictionary.word2index)

        # Ask dictionary to prepare negative samples before creating batches
        self.dictionary.prepare_negative_sampling_table()

        # Create batches
        self.pairs = self.collection_to_pairs()
        shuffle(self.pairs)
        self.batches = self.pairs_to_batches(enable_cuda)

        # Collect words in order for later use
        self.words = [self.dictionary.index2word[i] for i in range(len(self.dictionary.index2word))]

        logging.info("Tokenized all data, constructed string pairs, " +
                     "initialized vocabulary.")

    def remove_uncommon_from_list(self, commons, sentence):
        return [x if x in commons else "UNK"for x in sentence]


    def collection_to_pairs(self):
        """Create training pairs for the entire collection.
        Returns:
            list of tuples of lenght three
        """
        pairs = []
        for i, sentence in enumerate(self.lines):
            indices = self.to_indices(sentence)
            for j, centre in enumerate(indices):
                pre = indices[max(j - self.window, 0):j]
                post = indices[j+1:min(j + self.window + 1, len(indices))]
                pre = (self.window - len(pre)) * [self.dictionary.word2index["<s>"]] + pre
                post = post + (self.window - len(post)) * [self.dictionary.word2index["</s>"]]
                context = pre + post
                negative = self.dictionary.sample([centre] + context, self.window * 2)

                # Add every centre and context as a pair
                pairs.append((centre, context, negative))

        logging.info("Initialized numerical training pairs.")
        return pairs

    def pairs_to_batches(self, enable_cuda):
        """Create batches out of pairs of (neighbour, centre, negative samples).

        Args:
            enable_cuda (bool): cuda batches or not
        Returns:
            list of batches
        """
        batches = []
        batch_centre = torch.LongTensor(self.batch_size)
        batch_context = torch.LongTensor(self.batch_size, self.window * 2)
        batch_negative = torch.LongTensor(self.batch_size, self.window * 2)
        
        # Go through data in steps of batch size
        for i in range(0, len(self.pairs) - self.batch_size, self.batch_size):
            batch_pairs = self.pairs[i:i+self.batch_size]
            for j, (centre, context, negative) in enumerate(batch_pairs):
                batch_centre[j] = centre
                batch_context[j, :] = torch.LongTensor(context)
                batch_negative[j, :] = torch.LongTensor(negative)
            
            a = Variable(batch_centre)
            b = Variable(batch_context)
            c = Variable(batch_negative)

            if enable_cuda:
                batches.append((a.cuda(), b.cuda(), c.cuda()))
            else:
                batches.append((a, b, c))
        return batches

    def prepare_test(self, sentences_path="../data/lst/lst_valid.preprocessed",
                     cand_path="../data/lst/lst.gold.candidates"):
        test_pairs = []
        candidates = dict()
        candidates_rest = dict()
        with open(cand_path, 'r') as f:
            for line in f:
                term, term_candidates = tuple(line.split("::"))
                term_candidates = term_candidates.strip().split(";")
                candidates[term] = self.to_indices(
                    [term for term in term_candidates if term in self.dictionary.word2index]
                )
                candidates_rest[term] = [term for term in term_candidates if term not in self.dictionary.word2index]

        with open(sentences_path, 'r') as f:
            for line in f:
                # Read in data line by line
                term, number, pos, sentence = tuple(line.split("\t"))
                pos = int(pos)
                sentence = sentence.split()

                # Extract the context of the term and pad with <s> or </s>
                indices = self.to_indices(sentence)
                if sentence[pos] not in self.dictionary.word2index:
                    c = self.word2index[term.split('.')[0]]
                else:
                    c = indices[pos]
                pre = indices[max(pos - self.window, 0):pos]
                post = indices[pos+1:min(pos + self.window + 1, len(indices))]
                #pre = (self.window - len(pre)) * [self.dictionary.word2index["<s>"]] + pre
                #post = post + (self.window - len(post)) * [self.dictionary.word2index["</s>"]]
                context = Variable(torch.LongTensor([pre + post]))
                centre = Variable(torch.LongTensor([c]))
                if self.enable_cuda:
                    test_pairs.append((centre.cuda(), context.cuda(), int(number), term, candidates[term], candidates_rest[term]))
                else:
                    test_pairs.append((centre, context, int(number), term, candidates[term], candidates_rest[term]))
        self.test_pairs = test_pairs
        return self.test_pairs

    def to_indices(self, sequence):
        """Represent a history of words as a list of indices.

        Args:
            sequence (list of stringss): text to turn into indices
        """
        return [self.dictionary.word2index[w] for w in sequence]

    def prepare(self, sequence):
        """Add start and end tags. Add words to the dictionary.

        Args:
            sequence (list of stringss): text to turn into indices
        """
        return ['<s>'] + sequence.split() + ['</s>']

if __name__ == '__main__':
    a = Corpus(sys.argv[1], 2, 10)