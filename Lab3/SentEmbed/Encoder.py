from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, embeddings, hidden_size, input_size, max_pos, type, enable_cuda):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.type = type
        self.enable_cuda = enable_cuda

        self.network = nn.LSTM(hidden_size, hidden_size, batch_first=True,
                               bidirectional=True)

        self.dropout_rate_0 = 0.5
        self.dropout_rate = 0.5

        self.embeddings = nn.Embedding(input_size, hidden_size)
        self.embeddings.weight.data.copy_(embeddings)
        self.embeddings.weight.requires_grad = False

        if self.type == "gran":
            self.bias = nn.Parameter(torch.randn(hidden_size))
            self.w_x = nn.Linear(hidden_size, hidden_size)
            self.w_h = nn.Linear(hidden_size, hidden_size)

    def gate(self, x, h):
        return x * F.sigmoid(self.w_x(x) + self.w_h(h) + self.bias)

    def forward(self, english, validation=False):
        """Forward pass for the encoder.

        Args:
            english (Variable LongTensor): indices of english words
            validation (bool): whether you are validating (don't use dropout)

        Returns:
            vocab_probs: distribution over vocabulary
            hidden: hidden state of decoder
        """
        hidden, c = self.init_hidden(english.shape[0], self.enable_cuda)
        english_embed = self.embeddings(english)

        # Don't apply dropout if validating
        if not validation:
            english_embed = F.dropout(english_embed, p=self.dropout_rate)

        output, (hidden, c) = self.network(english_embed, (hidden, c))
    
        if self.type == "gran":
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
            return torch.mean(self.gate(english_embed, output), dim=1).unsqueeze(0)
        else:
            return torch.add(hidden[0, :, :], hidden[1, :, :]).unsqueeze(0)

    def eval(self, english):
        """Evaluation pass for the encoder: don't apply dropout.

        Args:
            english (Variable LongTensor): indices of english words

        Returns:
            hidden: hidden state of decoder
        """
        return self.forward(english, True)

    def init_hidden(self, batch_size, enable_cuda):
        """Initialize the first hidden state randomly.

        Args:
            batch_size (int)
            enable_cuda (bool): whether a GPU is available

        Returns:
            Variable FloatTensor
        """
        if enable_cuda:
            return (Variable(torch.randn(2, batch_size, self.hidden_size)).cuda(), 
                    Variable(torch.randn(2, batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.randn(2, batch_size, self.hidden_size)),
                    Variable(torch.randn(2, batch_size, self.hidden_size)))