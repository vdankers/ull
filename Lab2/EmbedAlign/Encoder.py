from __future__ import unicode_literals, print_function, division
import torch
import logging
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encoder for Embed Align."""

    def __init__(self, vocab_size_e, dim, enable_cuda=False):
        """Initialize parameters."""
        super(Encoder, self).__init__()
        self.enable_cuda = enable_cuda
        self.dim = dim

        self.in_embeddings = nn.Embedding(vocab_size_e, dim)
        self.affine_mu1 = nn.Linear(dim*2, dim, bias=True)
        self.affine_mu2 = nn.Linear(dim, dim, bias=True)
        self.affine_sigma1 = nn.Linear(dim*2, dim, bias=True)
        self.affine_sigma2 = nn.Linear(dim, dim, bias=True)
        #self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, bias=True,
        #                    batch_first=True, dropout=0, bidirectional=True)

    def forward(self, english, valid=False):
        logging.basicConfig(level=logging.DEBUG)
        # Go from one hot to vectors
        
        english = self.in_embeddings(english)
        mean = torch.mean(english, dim=1)
        mean = mean.unsqueeze(1).repeat(1, english.shape[1], 1)
        h_summed = torch.cat((english, mean), dim=2)

        # Calculate mu and sigma
        mu = self.affine_mu1(h_summed)
        sigma = F.softplus(self.affine_sigma1(h_summed))
        return mu, sigma
