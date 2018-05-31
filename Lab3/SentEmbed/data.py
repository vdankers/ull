import nltk
import sys
import logging
import random
import torch
import pickle
import os
import numpy as np

from tqdm import tqdm
from nltk.tokenize import word_tokenize
from torch import LongTensor
from torch.autograd import Variable
from collections import defaultdict, Counter
from copy import deepcopy


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

    def to_unk(self):
        """From now on your dictionaries default to UNK for unknown words."""
        unk = self.add_word("UNK")
        self.word2index = defaultdict(lambda: unk, self.word2index)


class Corpus(object):
    """Collects words and corresponding associations, preprocesses them."""

    def __init__(self, pathl1, batch_size, min_count, lower, enable_cuda, embed_dict):
        """Initialize pairs of words and associations.

        Args:
            pathl1 (str): the path to the L1 data
            pathl2 (str): the path to the L2 data
            batch_size (int): int indicating the desired size for batches
            num_symbols (int): number of symbols for BPE
            min_count (int): word must occur at least min count times
            lower (bool): if true, sentences are lowercased
            enable_cuda (bool): whether to cuda the batches
        """
        self.batch_size = batch_size
        self.dict = Dictionary()
        for word in embed_dict:
            self.dict.add_word(word)
        self.lines = []
        self.enable_cuda = enable_cuda
        self.max_pos = 0
        self.lower = lower

        # Read in the corpus
        with open(pathl1, 'r', encoding='utf8') as f_eng:
            i = 0
            for line in f_eng:
                i += 1
                line = self.prepare(line, lower, embed_dict)
                self.lines.append(line)

        random.shuffle(self.lines)
        self.lines = self.lines[:100000]

        self.longest_english = max([len(e) for e in self.lines])
        self.vocab_size = len(self.dict.word2index)

        # Create batches
        self.batches = self.get_batches(enable_cuda)
        logging.info("Created Corpus.")
        for word in self.dict.word2index:
            if word not in embed_dict: print(word)

    def load_data(self, pathl1):
        lines_e = []
        lines_f = []
        with open(pathl1, 'r', encoding='utf8') as f_eng:
            for line_e in f_eng:
                line_e = self.prepare(line_e, self.lower, self.dict.word2index)
                lines_e.append(line_e)
        return lines_e
                     
    def get_batches(self, enable_cuda):
        """Create batches from data in class.

        Args:
            enable_cuda (bool): cuda batches or not

        Returns:
            list of batches
        """
        # Sort lines by the length of the English sentences
        sorted_lengths = [[len(x), x]
                           for x in self.lines]
        sorted_lengths.sort()
        
        batches = []
        
        # Go through data in steps of batch size
        for i in range(0, len(sorted_lengths) - self.batch_size, self.batch_size):
            max_english = max([x[0] for x in sorted_lengths[i:i+self.batch_size]])
            batch_english = LongTensor(self.batch_size, max_english)

            for j, (lenght, line) in enumerate(sorted_lengths[i:i+self.batch_size]):
                # Map words to indices and pad with EOS tag
                line = self.pad_list(
                    line, True, max_english, pad=self.dict.word2index['</s>']
                )

                batch_english[j,:] = LongTensor(line)

            batch_english = Variable(batch_english)

            if enable_cuda:
                batch_english = batch_english.cuda()

            batches.append(batch_english)
        random.shuffle(batches)
        return batches

    def word_positions(self, line):
        """"Get the positions corresponding to a sentence.

        Args:
            line (list of str)

        Returns:
            result (list of int)
        """
        result = []
        pos = 1
        for word in line:
            result.append(pos)
            if pos > self.max_pos: self.max_pos = pos
            if not (len(word) > 2 and word[-2:] == '@@'): pos += 1
        return result
        
    def pad_list(self, line, english, length, pad=0):
        """Pads list to a certain length
        
        Args:
            input (list): list to pad
            length (int): length to pad to
            pad (object): object to pad with
        """
        line = self.to_indices(line)
        return line + [pad] * max(0,length - len(line))
        
    def to_indices(self, sequence):
        """Represent a history of words as a list of indices.

        Args:
            sequence (list of stringss): text to turn into indices
        """
        return [self.dict.word2index[w] for w in sequence
                if w in self.dict.word2index]

    def prepare(self, sequence, lower, embed_dict):
        """Add start and end tags. Add words to the dictionary.

        Args:
            sequence (list of stringss): text to turn into indices
        """
        if lower: sequence = sequence.lower()
        sequence = sequence.split()
        sequence = [word for word in sequence if word in embed_dict]
        return ['<s>'] + sequence + ['</s>']

