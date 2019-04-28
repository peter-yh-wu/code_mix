#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

import os
import math
import time
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from lm import FNNLM, DualLSTM
from utils.data import *
from utils.model import *
from configs import *
from vocab import Vocab
from dataset import BilingualDataSet
from torch.utils.data import DataLoader
from log import init_logger


def calc_sent_loss(sent, model, criterion):
    """
    Calculate the loss value for the entire sentence
    """
    targets = torch.LongTensor([model.vocab[tok] for tok in sent[1:]]).to(DEVICE)
    logits = model(sent)
    loss = criterion(logits, targets)
    return loss


def generate_sent(model, max_len):
    """
    Generate a sentence
    """
    hist = ['<s>']
    eos = model.vocab['<s>']
    sent = []

    while len(sent) < max_len:
        logits = model(hist + ['<s>'])
        if logits.dim() > 1:
            logits = logits[-1]
        next_word = gumbel_argmax(logits, dim=0)
        if next_word == eos:
            break
        sent.append(model.vocab.itos[next_word])
        hist += [model.vocab.itos[next_word]]

    return sent


def calc_sentence_logprob(model, sentence):
    """
    Calculates the sentence log-prob
    """
    if len(sentence) < 1:
        return -float('inf')

    log_probs = torch.log(F.softmax(model(sentence), dim=0))
    ids = torch.Tensor(sentence[1:]).long()
    sentence_log_prob = torch.sum(log_probs.gather(1, ids.view(-1, 1)))

    return sentence_log_prob.item()

if __name__ == '__main__':
    # initialize logger
    logger = init_logger()
    logger.info(args)

    # Read in the data
    logger.info('Loading dataset...')
    dataset = read_dataset(args.data)
    dataset = dataset[: int(len(dataset) * args.subset)]
    train = dataset[: int(len(dataset)*0.8)]
    dev = dataset[int(len(dataset)*0.8) + 1: -1]
    vocab = Vocab(train)
    logger.info('  Training samples: {}'.format(len(train)))
    logger.info('  Dev samples:      {}'.format(len(dev)))
    logger.info('  Vocabulary size:  {}'.format(len(vocab)))

    # Initialize the model and the optimizer
    logger.info('Building model...')
    if args.model.lower() == 'lstm':
        model = DualLSTM(batch_size=args.batch, hidden_size=args.hidden,
                         embed_size=args.embed, n_gram=args.ngram,
                         vocab=vocab, vocab_size=len(vocab), dropout=args.dp,
                         embedding=None, freeze=False)
    elif args.model.lower() == 'fnn':
        model = FNNLM(n_words=len(vocab), emb_size=args.embed,
                      hid_size=args.hidden, num_hist=args.ngram, dropout=args.dp)
    else:
        raise NotImplemented

    model = model.to(DEVICE)

    # Construct loss function and Optimizer.
    criterion = torch.nn.CrossEntropyLoss()
    if args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mm)
    elif args.optim.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters())
    else:
        optimizer = torch.optim.Adadelta(model.parameters())

    train_loss_record = []
    valid_loss_record = []
    train_acc_record = []
    valid_acc_record = []
    best_valid_predictions = []
    test_predictions = []
    last_dev = 1e20
    best_dev = 1e20

    # Perform training
    for epoch in range(args.epoch):
        # shuffle training data
        random.shuffle(train)
        # set the model to training mode
        model.train()
        train_words, train_loss = 0, 0.0
        train_sents = 0
        start = time.time()
        for sent in train:
            # TODO: mean or sum loss?
            loss = calc_sent_loss(sent, model, criterion)
            train_loss += loss.data
            train_words += (len(sent) - 2)
            train_sents += 1
            optimizer.zero_grad()
            loss.backward()
            # TODO: add clip_grad?
            # clip_grad_norm helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            for p in model.parameters():
                p.data.add_(-args.lr, p.grad.data)
            optimizer.step()
            if train_sents % 500 == 0:
                logger.info("--finished %r sentences (sentence/sec=%.2f)"
                            % (train_sents, train_sents / (time.time() - start)))

            model.detach()

        logger.info("Epoch %r: train loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (
            epoch, train_loss / train_words, math.exp(train_loss / train_words),
            train_words / (time.time() - start)))

        train_loss_record.append(train_loss)

        # Evaluate on dev set
        # set the model to evaluation mode
        model.eval()
        dev_words, dev_loss = 0, 0.0
        start = time.time()
        with torch.no_grad():
            for sent in dev:
                # sentences = batch.to(DEVICE)
                loss = calc_sent_loss(sent, model, criterion)
                dev_loss += loss.data
                dev_words += (len(sent) - 2)

        # Keep track of the development accuracy and reduce the learning rate if it got worse
        if last_dev < dev_loss and hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate /= 2
        last_dev = dev_loss

        # Keep track of the best development accuracy, and save the model only if it's the best one
        if best_dev > dev_loss:
            if not os.path.exists('models'):
                try:
                    os.mkdir('models')
                except Exception as e:
                    print("Can not create models directory, %s" % e)
            torch.save(model.state_dict(), "{}/best.pt".format(args.models_dir))
            best_dev = dev_loss
        torch.save(model.state_dict(), "{}/epoch_{}.pt".format(args.models_dir, epoch))

        # Save the model
        logger.info("Epoch %r: dev loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (
            epoch, dev_loss / dev_words, math.exp(dev_loss / dev_words),
            dev_words / (time.time() - start)))

        # Generate a few sentences
        for _ in range(5):
            sentence = generate_sent(model, args.maxlen)
            logger.debug(" ".join([word for word in sentence]))