"""Prepare a corpus for definition modelling.

Tasks performed:
    - Generate a vocab dictionary that maps words to indices and vice versa.
    - Preprocess text by adding start and end tags.
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

    def to_unk(self):
        """From now on your dictionaries default to UNK for unknown words."""
        unk = self.add_word("UNK")
        self.word2index = defaultdict(lambda: unk, self.word2index)


class Corpus(object):
    """Collects words and corresponding associations, preprocesses them."""

    def __init__(self, pathl1, pathl2, batch_size, nr_docs, nr_unique_words, enable_cuda=False):
        """Initialize pairs of words and associations.

        Args:
            pathl1 (str): the path to the L1 data
            pathl2 (str): the path to the L2 data
            batch_size (int): int indicating the desired size for batches
            nr_docs (int): how many sentences should be used from the corpus
            enable_cuda (bool): whether to cuda the batches
        """
        self.batch_size = batch_size
        self.dictionary_e = Dictionary()
        self.dictionary_f = Dictionary()
        self.lines_e = []
        self.lines_f = []
        self.enable_cuda = enable_cuda

        # Read in the corpus
        with open(pathl1, 'r') as f_eng, open(pathl2, 'r') as f_fre:
            for line_e in f_eng:
                line_f = f_fre.readline()
                line_e = self.prepare(line_e)
                line_f = self.prepare(line_f)
                self.dictionary_e.add_text(line_e)
                self.dictionary_f.add_text(line_f)
                self.lines_e.append(line_e)
                self.lines_f.append(line_f)
                if len(self.lines_e) == nr_docs and nr_docs != -1:
                    break

        most_common_f = set([x[0] for x in self.dictionary_f.counts.most_common(nr_unique_words)])
        most_common_e = set([x[0] for x in self.dictionary_e.counts.most_common(nr_unique_words)])                   
        
        # Redo, but remove infrequent words
        dictionary_norare_e = Dictionary()
        dictionary_norare_f = Dictionary()
        for i, line in enumerate(self.lines_e):
            new_line = self.remove_uncommon_from_list(most_common_e, line)
            dictionary_norare_e.add_text(new_line)
            self.lines_e[i] = new_line
        for i, line in enumerate(self.lines_f):
            new_line = self.remove_uncommon_from_list(most_common_f, line)
            dictionary_norare_f.add_text(new_line)
            self.lines_f[i] = new_line
        self.dictionary_e = dictionary_norare_e
        self.dictionary_f = dictionary_norare_f
        self.dictionary_e.to_unk()
        self.dictionary_f.to_unk()
        self.vocab_size_e = len(self.dictionary_e.word2index)
        self.vocab_size_f = len(self.dictionary_f.word2index)

        # Create batches
        self.batches = self.get_batches(enable_cuda)

        logging.info("Created Corpus.")
                     
    def remove_uncommon_from_list(self, commons, sentence):
        return [x if x in commons else "UNK"for x in sentence]
                     
    def get_batches(self, enable_cuda):
        """Create batches from data in class.

        Args:
            enable_cuda (bool): cuda batches or not
        Returns:
            list of batches
        """
        
        lines_with_lengths = [[len(x), len(y), x, y] for x,y in zip(self.lines_e, self.lines_f)]
        lines_with_lengths.sort()
        eng_sents = [x[2] for x in lines_with_lengths]
        fre_sents = [x[3] for x in lines_with_lengths]

        batches = []
        
        # Go through data in steps of batch size
        for i in range(0, len(eng_sents) - self.batch_size, self.batch_size):
            longest_french_sentence = max([x[1] for x in lines_with_lengths[i:i+self.batch_size]])
            longest_english_sentence = max([x[0] for x in lines_with_lengths[i:i+self.batch_size]])
            batch_french = torch.LongTensor(self.batch_size, longest_french_sentence)
            batch_english = torch.LongTensor(self.batch_size, longest_english_sentence)

            batch_lines_e = eng_sents[i:i+self.batch_size]
            batch_lines_f = fre_sents[i:i+self.batch_size]
            for j, (eline, fline) in enumerate(zip(batch_lines_e, batch_lines_f)):
                fline = self.pad_list(self.to_indices(fline, False), longest_french_sentence, pad = self.dictionary_f.word2index['</s>'])
                eline = self.pad_list(self.to_indices(eline, True), longest_english_sentence, pad = self.dictionary_e.word2index['</s>'])
                
                batch_french[j, :] = torch.LongTensor(fline)
                batch_english[j,:] = torch.LongTensor(eline)
            if enable_cuda:
                batches.append((Variable(batch_english).cuda(), Variable(batch_french).cuda()))
            else:
                batches.append((Variable(batch_english), Variable(batch_french)))
        random.shuffle(batches)
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
                    [term for term in term_candidates if term in self.dictionary_e.word2index]
                )
                candidates_rest[term] = [term for term in term_candidates if term not in self.dictionary_e.word2index]

        with open(sentences_path, 'r') as f:
            for line in f:
                # Read in data line by line
                term, number, pos, sentence = tuple(line.split("\t"))
                pos = int(pos)
                sentence = sentence.split()

                # Extract the context of the term
                indices = self.to_indices(sentence, True)
                pre = indices[:pos]
                post = indices[pos+1:]              
                centre_tensor = Variable(torch.LongTensor([pre + [indices[pos]] + post]))
                if self.enable_cuda:
                    centre_tensor = centre_tensor.cuda()

                # Create tensors with the candidates for replacements
                candidate_tensors = []
                for candidate in candidates[term]:
                    tensor = Variable(torch.LongTensor([pre + [candidate] + post]))
                    if self.enable_cuda:
                        tensor = tensor.cuda()
                    candidate_tensors.append((candidate, tensor))
                
                test_pairs.append((centre_tensor, int(number), term, candidate_tensors, candidates_rest[term]))
        self.test_pairs = test_pairs
        return self.test_pairs

    def pad_list(self, input, length, pad=0):
        """Pads list to a certain length
        
        Args:
            input (list): list to pad
            length (int): length to pad to
            pad (object): object to pad with
        """
        return input + [pad] * max(0,length - len(input))
        
    def to_indices(self, sequence, english=True):
        """Represent a history of words as a list of indices.

        Args:
            sequence (list of stringss): text to turn into indices
        """
        if english:
            return [self.dictionary_e.word2index[w] for w in sequence]
        else:
            return [self.dictionary_f.word2index[w] for w in sequence]
    def prepare(self, sequence):
        """Add start and end tags. Add words to the dictionary.

        Args:
            sequence (list of stringss): text to turn into indices
        """
        return ['<s>'] + sequence.split() + ['</s>']

if __name__ == '__main__':
    l1path = './../data/hansards/training.en'
    l2path = './../data/hansards/training.fr'
    a = Corpus(l1path, l2path, 10, 2000, 10000)
    print(a.batches[80][1][8])
    print(a.batches[0][0].shape)
    for batch in a.batches:
        print(batch[0].shape, batch[1].shape)
        