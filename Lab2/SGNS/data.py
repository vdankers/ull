"""Prepare a corpus for training the skipgram negative sampling model.

Tasks performed:
    - Generate a vocab dictionary that maps words to indices and vice versa.
    - Preprocess text by adding start and end tags.
    - Read text files and save pairs centre - neighbour - negative samples.
    - Convert pairs to numbers in preparation of training a language model.
    - Calculate common phrases from a corpus.
    - Convert data pairs to batches that can be used to train a model in Pytorch.
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
from torch.autograd import Variable
from collections import defaultdict, Counter


class Dictionary(object):
    """Object that creates and keeps word2index and index2word dicts."""

    def __init__(self):
        """Initialize word - index mappings."""
        self.word2index = defaultdict(lambda: len(self.word2index))
        self.index2word = dict()
        self.counts = Counter()
        self.phrase_counts = Counter()

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
        for i, word in enumerate(text):
            self.add_word(word)
            if i != len(text) - 1:
                self.phrase_counts[(text[i], text[i + 1])] += 1

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
            neg_index = pos_index
            while neg_index is pos_index:
                neg_index = self.table[random.randint(0, len(self.table)-1)]
            samples.append(neg_index)
        return samples

    def to_unk(self):
        """From now on your dictionaries default to UNK for unknown words."""
        unk = self.add_word("UNK")
        self.word2index = defaultdict(lambda: unk, self.word2index)


class Corpus(object):
    """Collects words and corresponding associations, preprocesses them."""

    def __init__(self, path, window, min_count, batch_size, nr_docs,
                 neg_samples, enable_cuda=False):
        """Initialize pairs of words and associations.

        Args:
            path (str): file path to read data from
            window (int): window size for corpus pairs
            min_count (int): if words occur less often than this, remove them
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

        # Read in the corpus
        with open(path, 'r', encoding='Latin-1') as f:
            for line in f:
                line = self.prepare(line)
                self.dictionary.add_text(line)
                self.lines.append(line)
                if len(self.lines) == nr_docs and nr_docs != -1:
                    break

        self.lines = self.remove_min_count(min_count, self.lines)

        # Recreate your dictionary with the adapted corpus
        dictionary_norare = Dictionary()
        for i, line in enumerate(self.lines):
            dictionary_norare.add_text(line)
        self.dictionary = dictionary_norare

        # If new words are asked for their index, give UNK
        self.dictionary.to_unk()
        self.vocab_size = len(self.dictionary.word2index)

        # Ask dictionary to prepare negative samples before creating batches
        self.dictionary.prepare_negative_sampling_table()

        # Create batches
        self.pairs = self.collection_to_pairs()
        self.batches = self.pairs_to_batches(enable_cuda)

        # Collect words in order for later use
        self.words = self.dictionary.words

        logging.info("Tokenized all data, constructed string pairs, " +
                     "initialized vocabulary.")

    def downsample(self, table, lines):
        """Subsample frequent words.

        Args:
            table (dict): gives probability per word
            lines: list of sentences from the corpus

        Returns:
            list of sentences from the corpus, downsampled
        """
        for i, line in enumerate(lines):
            new_line = []
            for j, word in enumerate(line):
                if random.random() >= table[word]:
                    new_line.append(word)
            lines[i] = new_line
        logging.info("Subsampling frequent words completed.")
        return lines

    def replace_phrases(self, phrases, lines):
        """Connect common phrases in the corpus by an underscore to make them
        one word.

        Args:
            phrases (list): list of tuples of the phrase pairs
            lines: list of sentences from the corpus

        Returns:
            list of sentences from the corpus, with phrases replaced
        """
        for i, line in enumerate(lines):
            new_line = []
            phrase_found = False
            for j, word in enumerate(line):
                # If it's a phrase, combine words
                if tuple(line[j:j+2]) in phrases:
                    new_line.append("{}_{}".format(line[j], line[j+1]))
                    phrase_found = True
                # Else just add the word
                else:
                    if not phrase_found: new_line.append(word)
                    phrase_found = False
            lines[i] = new_line
        logging.info("Replacing phrases completed.")
        return lines

    def remove_min_count(self, min_count, lines):
        """Remove words occurring less than a minimum count.

        Args:
            min_count (int)
            lines (list): list of sentences from the corpus

        Returns:
            list: sentences with the words occurring < min_count removed
        """
        for i, line in enumerate(lines):
            new_line = []
            for word in line:
                if self.dictionary.counts[word] >= min_count:
                    new_line.append(word)
            lines[i] = new_line
        logging.info("Removing min counts completed.")
        return lines

    def collection_to_pairs(self):
        """Create training pairs for the entire collection.

        Returns:
            list of tuples of length three
        """
        pairs = []
        for i, sentence in enumerate(self.lines):
            indices = self.to_indices(sentence)
            for j, centre in enumerate(indices):
                context = indices[max(j - self.window, 0):j] + indices[j+1:min(j + self.window + 1, len(indices))]
                for neighbour in context:
                    # Add every centre and neighbour as a pair, along with negative samples from the dict
                    pairs.append((neighbour, centre, self.dictionary.sample(centre, self.neg_samples)))

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
        batch_neighbour = torch.LongTensor(self.batch_size)
        batch_centre = torch.LongTensor(self.batch_size)
        batch_neg_samples = torch.LongTensor(self.batch_size, self.neg_samples)
        
        # Go through data in steps of batch size
        for i in range(0, len(self.pairs) - self.batch_size, self.batch_size):
            batch_pairs = self.pairs[i:i+self.batch_size]
            for j, (neighbour, centre, neg_samples) in enumerate(batch_pairs):
                batch_neighbour[j] = neighbour
                batch_centre[j] = centre
                batch_neg_samples[j, :] = torch.LongTensor(neg_samples)
            if enable_cuda:
                batches.append((Variable(batch_neighbour).cuda(),
                                Variable(batch_centre).cuda(),
                                Variable(batch_neg_samples).cuda()))
            else:
                batches.append((Variable(batch_neighbour),
                                Variable(batch_centre),
                                Variable(batch_neg_samples)))
        return batches


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
