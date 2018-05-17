from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encoder for Bayesian Skip-Gram."""

    def __init__(self, vocab_size, dim, dim2, window, enable_cuda=False):
        """Initialize parameters."""
        super(Encoder, self).__init__()
        self.enable_cuda = enable_cuda
        self.window = window * 2

        self.in_embeddings = nn.Embedding(vocab_size, dim)
        self.M = nn.Linear(dim * 2, dim, bias=False)
        self.U = nn.Linear(dim, dim, bias=True)
        self.W = nn.Linear(dim, dim, bias=True)

    def forward(self, centre, context, valid=False):
        # Go from one hot to vectors
        centre = self.in_embeddings(centre).unsqueeze(1).repeat(1, context.shape[1], 1)
        context = self.in_embeddings(context)

        # Concatenate the centre word to every neighbour, apply relu and sum
        h = torch.cat((context, centre), dim=2)
        h = torch.sum(F.relu(self.M(h)), dim=1)
        
        # Calculate mu and sigma
        mu = self.U(h)
        sigma = F.softplus(self.W(h))
        return mu, sigma
