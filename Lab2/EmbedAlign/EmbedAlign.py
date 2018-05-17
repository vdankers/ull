from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class EmbedAlign(nn.Module):
    """EmbedAlign architecture. (techinically most of the code is still SGNS)"""

    def __init__(self, vocab_size, dim, enable_cuda=False):
        """Initialize parameters."""
        super(EmbedAlign, self).__init__()
        self.enable_cuda = enable_cuda

        # Keep embeddings fixed, don't adapt the weights
        # Cuda the embeddings for speedup
        self.in_embeddings = nn.Embedding(vocab_size, dim)
        self.out_embeddings = nn.Embedding(vocab_size, dim)
        # self.h_out = nn.Linear(dim, vocab_size, bias=True)

    def forward(self, centre, neighbour, neg_samples):
        """Calculate hidden state and log probabilities per vocabulary word."""

        # Calculate the loss for the positive sample
        losses = []
        centre = self.in_embeddings(centre)
        neighbour = self.out_embeddings(neighbour)
        # For the dot product, we use element wise multiplication and then summing
        score = torch.sum(centre * neighbour, dim=1)
        losses.append(torch.sum(F.logsigmoid(score)))

        # Calculate the loss for the negative samples
        neg_samples = self.out_embeddings(neg_samples).transpose(1, 2)
        score = torch.sum(centre.unsqueeze(2) * neg_samples, dim=1)
        losses.append(torch.sum(torch.sum(F.logsigmoid(-1 * score))))
        return -1 * sum(losses)