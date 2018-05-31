from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging

from torch.distributions.multivariate_normal import MultivariateNormal


class Decoder(nn.Module):
    """Decoder for Bayesian Skip-Gram."""

    def __init__(self, vocab_size, dim, dim2, window, batch_size, enable_cuda=False):
        """Initialize parameters."""
        super(Decoder, self).__init__()
        self.enable_cuda = enable_cuda
        self.window = window * 2
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dim = dim

        self.C = nn.Embedding(vocab_size, dim)
        self.L = nn.Embedding(vocab_size, dim)
        self.S = nn.Embedding(vocab_size, dim)
        self.affine = nn.Linear(dim, vocab_size)

    def KL(self, mu_x, sigma_x, mu_l, sigma_l, no_sum=False):
        a = torch.log(torch.div(sigma_x, sigma_l))
        b = torch.div((torch.pow(sigma_l, 2) + torch.pow((mu_l - mu_x), 2)), (2 * torch.pow(sigma_x, 2)))
        c = a + b
        if no_sum:
            return c.sub(0.5)
        else:
            KL = torch.sum(c.sub(0.5))
            if KL.item() < 0:
                logging.warning("Your KL < 0")
            return KL

    def forward(self, mu_l, sigma_l, centre, context, batch_size=0):
        # Likelihood term of the loss
        noise = Variable(torch.randn(sigma_l.shape), requires_grad=False)
        if self.enable_cuda: noise = noise.cuda()
        z = mu_l + sigma_l * noise
        probs = F.softmax(self.affine(z), dim=1)
        context_probs = torch.gather(probs, 1, context)
        likelihood = torch.sum(torch.log(context_probs))

        # Kullback Leibler
        mu_x = self.L(centre)
        sigma_x = F.softplus(self.S(centre))
        return likelihood, self.KL(mu_x, sigma_x, mu_l, sigma_l)
