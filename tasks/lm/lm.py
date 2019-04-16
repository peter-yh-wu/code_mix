#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

import torch
import torch.nn as nn

from configs import DEVICE
from utils.data import is_english_word
from utils.model import weight_init


class FNNLM(nn.Module):
    """
    Feed-forward Neural Network Language Model
    """
    def __init__(self, n_words, emb_size, hid_size, num_hist, dropout):
        super(FNNLM, self).__init__()
        self.embedding = nn.Embedding(n_words, emb_size)
        self.fnn = nn.Sequential(
            nn.Linear(num_hist*emb_size, hid_size), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_size, n_words)
        )

    def forward(self, words):
        emb = self.embedding(words)       # 3D Tensor of size [batch_size x num_hist x emb_size]
        feat = emb.view(emb.size(0), -1)  # 2D Tensor of size [batch_size x (num_hist*emb_size)]
        logit = self.fnn(feat)            # 2D Tensor of size [batch_size x nwords]

        return logit


class DualLSTM(nn.Module):
    """
    Dual LSTM Language Model
    """
    def __init__(self, batch_size, hidden_size, embed_size, n_gram, vocab, vocab_size,
                 dropout=0.5, embedding=None, freeze=False):
        super(DualLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.vocab_size = vocab_size
        if embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings=embedding, freeze=freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        self.dummy_tok = torch.zeros((1, embed_size)).to(DEVICE)

        self.lstm_en = nn.LSTMCell(input_size=embed_size*n_gram, hidden_size=hidden_size, bias=False)
        self.lstm_cn = nn.LSTMCell(input_size=embed_size*n_gram, hidden_size=hidden_size, bias=False)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(vocab_size, vocab_size)
        )

        # [batch_size, hidden_size]
        self.hidden_en = self.init_hidden()
        # self.cell_en = self.init_hidden()
        self.hidden_cn = self.init_hidden()
        # self.cell_cn = self.init_hidden()
        self.cell = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size).to(DEVICE)

    def detach(self):
        self.hidden_en.detach_()
        self.hidden_cn.detach_()
        self.cell.detach_()

    def init_weights(self):
        self.apply(weight_init)

    def forward(self, sentence):
        sent_embed, embed_mask = self.embed_sentence(sentence)
        lstm_out = []
        for i in range(len(sent_embed)):
            if embed_mask[i] > 0:
                self.hidden_en, self.cell = self.lstm_en(sent_embed[i], (self.hidden_en, self.cell))
                self.hidden_cn, self.cell = self.lstm_cn(self.dummy_tok, (self.hidden_cn, self.cell))
            else:
                self.hidden_cn, self.cell = self.lstm_cn(sent_embed[i], (self.hidden_cn, self.cell))
                self.hidden_en, self.cell = self.lstm_en(self.dummy_tok, (self.hidden_en, self.cell))
            lstm_out.append(self.hidden_en + self.hidden_cn)
        lstm_out = torch.stack(lstm_out)

        prediction = self.fc(lstm_out.squeeze(1))
        return prediction

    def embed_sentence(self, sentence):
        embedding = []
        embed_mask = torch.zeros(len(sentence))
        for idx, token in enumerate(sentence[:-1]):
            if is_english_word(token) or token in ['<pad>', '<unk>', '<s>', '</s>']:
                try:
                    embedding.append(self.embedding(torch.LongTensor([self.vocab.stoi[token]]).to(DEVICE)))
                    embed_mask[idx] = 1.
                except Exception:
                    print(sentence, self.vocab_size, token, self.vocab.stoi[token])
            else:
                try:
                    embedding.append(self.embedding(torch.LongTensor([self.vocab.stoi[token]]).to(DEVICE)))
                except Exception:
                    print(sentence, self.vocab_size, token, self.vocab.stoi[token])
        return torch.stack(embedding).to(DEVICE), embed_mask.to(DEVICE)
