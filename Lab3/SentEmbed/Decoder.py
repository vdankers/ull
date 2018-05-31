import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class Decoder(nn.Module):
    def __init__(self, embeddings, hidden_size, vocab_size, end_token, max_length):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.end_token = end_token
        self.type = type

        # Initialize dropout rates
        self.dropout_rate_0 = 0.5
        self.dropout_rate = 0.5
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding.weight.data.copy_(embeddings)
        self.embedding.weight.requires_grad = False
        self.network = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, token, hidden, c, validation=False):
        """Forward pass for the GRU decoder with attention.

        Args:
            token (Variable LongTensor): token index
            hidden (Variable FloatTensor): hidden state from encoder
            c (Variable FloatTensor): memory state of LSTM
            validation (bool): whether we are evaluating (do not apply dropout)

        Returns:
            vocab_probs: distribution over vocabulary
            hidden: hidden state of decoder
        """
        # Apply attention to English sentence
        token = self.embedding(token).unsqueeze(1)

        # Only apply dropout during training
        if not validation:
            hidden = F.dropout(hidden, p=self.dropout_rate)
        output, (hidden, c) = self.network(token, (hidden, c))

        if not validation:
            output_over_vocab = self.out(F.dropout(output[:, 0, :], p=self.dropout_rate))
        else:
            output_over_vocab = self.out(output[:, 0, :])
        vocab_probs = self.logsoftmax(output_over_vocab)
        return vocab_probs, hidden, c

    def eval(self, token, hidden, c):
        """Evaluation pass for the GRU decoder with attention: don't apply
        dropout.

        Args:
            token (Variable LongTensor): token indices batchwise
            english (Variable FloatTensor): encoding of english sentence
            hidden (Variable FloatTensor): hidden state from encoder
            c (Variable FloatTensor): memory state of LSTM

        Returns:
            vocab_probs: distribution over vocabulary
            hidden: hidden state of decoder
        """
        probs, hidden, c = self.forward( token, hidden, c, True) 
        return probs, hidden, c

    def init_hidden(self, batch_size, enable_cuda):
        """Initialize the first hidden state randomly.

        Args:
            batch_size (int)
            enable_cuda (bool): whether a GPU is available

        Returns:
            Variable FloatTensor
        """
        if enable_cuda:
            return Variable(torch.randn(1, batch_size, self.hidden_size)).cuda(), Variable(torch.randn(1, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(1, batch_size, self.hidden_size)), Variable(torch.randn(1, batch_size, self.hidden_size))