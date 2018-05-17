from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class SGNS(nn.Module):
    """Skip-Gram Negative Sampling architecture."""

    def __init__(self, vocab_size, dim, enable_cuda=False):
        """Initialize parameters.

        Args:
            vocab_size: size of vocabulary
            dim: desired number of dimensions
            enable_cuda: whether GPU is available
        """
        super(SGNS, self).__init__()
        self.enable_cuda = enable_cuda
        self.in_embeddings = nn.Embedding(vocab_size, dim)
        self.out_embeddings = nn.Embedding(vocab_size, dim)

    def forward(self, centre, neighbour, neg_samples):
        """Calculate hidden state and log probabilities per vocabulary word.

        Args:
            centre: indices of centre words
            neighbour: indices of context words
            neg_samples; indices of negative samples
        """

        # Calculate the loss for the positive sample
        losses = []
        centre = self.in_embeddings(centre)
        neighbour = self.out_embeddings(neighbour)

        # For the dot product, we use element wise multiplication and summing
        score = torch.sum(centre * neighbour, dim=1)
        losses.append(torch.sum(F.logsigmoid(score)))

        # Calculate the loss for the negative samples
        neg_samples = self.out_embeddings(neg_samples).transpose(1, 2)
        score = torch.sum(centre.unsqueeze(2) * neg_samples, dim=1)
        losses.append(torch.sum(torch.sum(F.logsigmoid(-1 * score))))
        return -1 * sum(losses)