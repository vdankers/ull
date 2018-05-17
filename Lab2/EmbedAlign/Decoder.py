from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging
import numpy as np


class Decoder(nn.Module):
    """Decoder for Bayesian Skip-Gram."""

    def __init__(self, vocab_size_e, vocab_size_f, dim, batch_size, enable_cuda=False):
        """Initialize parameters."""
        super(Decoder, self).__init__()
        self.enable_cuda = enable_cuda
        self.batch_size = batch_size
        print(vocab_size_e, vocab_size_f)

        self.affine1 = nn.Linear(dim, vocab_size_e)
        self.affine2 = nn.Linear(dim, vocab_size_f)

    def KL(self, mu_x, sigma_x, mu_l, sigma_l):
        a = torch.log(torch.div(sigma_x, sigma_l))
        b = torch.div((torch.pow(sigma_l, 2) + torch.pow((mu_l - mu_x), 2)), (2 * torch.pow(sigma_x, 2)))
        c = a + b
        KL = torch.sum(c.sub(0.5))
        if KL.data[0] < 0:
            logging.warning("Your KL < 0")
        return KL

    def forward(self, mu_l, sigma_l, english, french, getting_aer = False):
        """Forward pass through the generative part of the network"""
        noise = Variable(torch.randn(sigma_l.shape), requires_grad=False)
        if self.enable_cuda:
            noise = noise.cuda()
        z = mu_l + sigma_l

        probs_e = F.softmax(self.affine1(z), dim=2)
        probs_f = F.softmax(self.affine2(z), dim=2)

        if getting_aer: # this was put in at the end, so doesn't really fit the control flow, but here we are
            probs_f = probs_f.data.cpu().numpy()
            best_alignments = []
            for j in list(french[0,:]):
                data = probs_f[0, :, int(j)]
                best_alignments.append(int(np.argmax(data)))
                #print(data, int(np.argmax(data)))
            return best_alignments
        
        # Select probs for log likelihood English part
        english_ll = 0
        french_ll = 0
        selected_probs_e = Variable(torch.FloatTensor(probs_e.shape[0], probs_e.shape[1]))
        if self.enable_cuda:
            selected_probs_e = selected_probs_e.cuda()
        for i in range(self.batch_size):
            for j in range(english.shape[1]):
                selected_probs_e[i, j] = torch.index_select(probs_e[i, j, :], 0, english[i, j])


                
        # Select probs for log likelihood French part
        selected_probs_f = Variable(torch.FloatTensor(probs_e.shape[0], french.shape[1]))
        if self.enable_cuda:
            selected_probs_f = selected_probs_f.cuda()
        for i in range(self.batch_size):
            columns = torch.index_select(probs_f[i, :, :], 1, french[i, :])
            selected_probs_f[i, :] = torch.mean(columns, dim=0)

        likelihood = torch.add(torch.sum(torch.log(selected_probs_e)), torch.sum(torch.log(selected_probs_f)))

        # Kullback Leibler
        mu_normal = Variable(torch.zeros(mu_l.shape), requires_grad=False)
        sigma_normal = Variable(torch.ones(sigma_l.shape), requires_grad=False)
        if self.enable_cuda:
            mu_normal = mu_normal.cuda()
            sigma_normal = sigma_normal.cuda()
        kl = self.KL(mu_normal, sigma_normal, mu_l, sigma_l)
        return likelihood, kl
