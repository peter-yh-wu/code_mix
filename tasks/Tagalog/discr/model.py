'''
Models for Discriminator Experiment

Peter Wu
peterw1@andrew.cmu.edu
'''

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.distributions.bernoulli import Bernoulli

from model_utils import *

class LSTMDiscriminator(nn.Module):
    '''
    As described in: https://arxiv.org/abs/1810.11895
    '''
    def __init__(self, vocab_size, word_dropout=0.2, emb_dim=300, hidden_dim=650):
        super(LSTMDiscriminator, self).__init__()
        self.prob_keep = 1-word_dropout
        self.emb_mat = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True, dropout=0.35, bidirectional=True)
        # w # a learned vector, want w dot h in forward

    def forward_repr(self, x):
        '''
        Args:
            LongTensor with shape (batch_size, seq_len)
        '''
        x_emb = emb_mat(x) # shape: (batch_size, seq_len, emb_dim)
        rw = Bernoulli(self.prob_keep).sample((x_emb.shape[1], ))
        x_emb = x_emb[:, rw==1] # (batch_size, new_seq_len, emb_dim)
        _, (h, ) = self.rnn(x_emb).squeeze() # (batch_size, 2*hidden_dim)
        return h

    def forward_score(self, x):
        '''
        Args:
            LongTensor with shape (batch_size, seq_len)
        '''
        h = forward_repr(x)
        return 0
        # TODO

    def forward(self, x):
        return forward_score(x)

class WERDiscriminatorLoss(nn.Module):
    def __init__(self):
        super(WERDiscriminatorLoss, self).__init__()

    def forward(self, x1, x2):
        return 0
        # TODO

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, args):
        super(LSTMLM, self).__init__()
        self.prob_keep = 1-args.word_dropout
        self.emb_mat = nn.Embedding(vocab_size, args.emb_dim)
        self.rnn = nn.LSTM(args.emb_dim, args.hidden_dim, batch_first=True, dropout=args.rnn_dropout, bidirectional=True)
        self.char_projection = nn.Sequential(
            nn.Linear(args.hidden_dim, vocab_size)
        )
        self.force_rate = args.teacher_force_rate

    def forward_step(self, prev_char, prev_hidden):
        '''
        Args:
            prev_char: shape (batch_size,)
            prev_hidden: pair of tensors with shape (1, batch_size, hidden_dim)
        
        Return:
            logits: tensor with shape (batch_size, vocab_size)
            y_pred: tensor with shape (batch_size,)
            new_hidden: pair of tensors with shape (1, batch_size, hidden_dim)
        '''
        emb = self.emb_mat(prev_char)
        out, new_hidden = self.rnn(emb, prev_hidden)
        logits = self.char_projection(out)
        y_pred = torch.max(logits, 1)[1]
        return logits, y_pred, new_hidden

    def forward(self, x, lens):
        '''
        Args:
            x: tensor with shape (batch_size, maxlen)
            lens: tensor with shape (batch_size,), comprised of
                length of each input sequence in batch
        '''
        maxlen, batch_size = x.shape
        mask = Variable(output_mask(maxlen, utterance_lengths).transpose(0, 1)).float()
            # shape: (batch_size, maxlen)

        hidden = None
        all_logits = []
        y_preds = []
        for i in range(maxlen):
            if len(y_preds) > 0 and self.force_rate < 1 and self.training:
                forced_char = x[:, i]
                gen_char = y_preds[-1]
                force_mask = Variable(forced_char.data.new(*forced_char.size()).bernoulli_(self.force_rate))
                char = (force_mask * forced_char) + ((1 - force_mask) * gen_char)
            else:
                char = x[:, i]
            logits, y_pred, hidden = self.forward_step(char, hidden)
            all_logits.append(logits)
            y_preds.append(y_pred)
        all_logits = torch.stack(all_logits, 1) # shape: (batch_size, maxlen, vocab_size)
        y_preds = torch.stack(y_preds, 1) # shape: (batch_size, maxlen)
        return all_logits, y_preds
