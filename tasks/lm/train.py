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

from torch.autograd import Variable
from lm import FNNLM, DualLSTM
from utils.data import *
from configs import *
from vocab import Vocab
from dataset import BilingualDataSet
from torch.utils.data import DataLoader
from log import init_logger


def calc_sent_loss(sent, model, criterion):
    """
    Calculate the loss value for the entire sentence
    """
    targets = torch.LongTensor([model.vocab[tok] for tok in sent + ['<s>']]).to(DEVICE)
    logits = model(['<s>'] + sent + ['<s>'])
    loss = criterion(logits, targets)

    return loss


def generate_sent(model):
    """
    Generate a sentence
    """
    hist = [model.vocab.itos[torch.randint(low=0, high=len(model.vocab), size=(1,), dtype=torch.int32)]]
    # hist += ['<s>']
    eos = model.vocab['<s>']
    while True:
        logits = model(hist + ['<s>'])[-1]
        prob = F.softmax(logits, dim=0)
        # next_word = prob.multinomial(1).data[0, 0]
        next_word = torch.argmax(prob)
        if next_word == eos or len(hist) == args.maxlen:
            break
        hist.append(model.vocab.itos[next_word])

    return hist


if __name__ == '__main__':
    logger = init_logger()
    logger.info(args)
    # Read in the data
    logger.info('Loading dataset...')
    train = read_dataset("SEAME-dev-set/dev_man/text")
    dev = read_dataset("SEAME-dev-set/dev_sge/text")
    vocab = Vocab(train)
    # train_set = BilingualDataSet(vocab, examples=train, padding=False, sort=False)
    # dev_set = BilingualDataSet(vocab, examples=dev, padding=False, sort=False)
    # datasets = {'train': train_set, 'dev': dev_set}
    # data_loaders = {
    #     name: DataLoader(
    #         dataset,
    #         batch_size=args.batch,
    #         shuffle=(name == 'train'),
    #         num_workers=args.nworkers,
    #         collate_fn=dataset.collate
    #     )
    #     for name, dataset in datasets.items()}

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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
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
        # shulle training data
        random.shuffle(train)
        # set the model to training mode
        model.train()
        train_sents, train_loss = 0, 0.0
        start = time.time()
        for sent in train:
            loss = calc_sent_loss(sent, model, criterion).mean()
            train_loss += loss.data
            train_sents += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if train_sents % 500 == 0:
                logger.info("--finished %r sentences (sentence/sec=%.2f)"
                            % (train_sents, train_sents / (time.time() - start)))
                # Generate a few sentences
                logger.info("Generate some sentences...")
                for _ in range(3):
                    sentence = generate_sent(model)
                    logger.debug(" ".join([word for word in sentence]))

            model.detach()

        logger.info("iter %r: train loss/word=%.4f, ppl=%.4f (sentence/sec=%.2f)" % (
            epoch, train_loss / train_sents, math.exp(train_loss / train_sents),
            train_sents / (time.time() - start)))

        train_loss_record.append(train_loss)

        # Evaluate on dev set
        # set the model to evaluation mode
        model.eval()
        dev_words, dev_loss = 0, 0.0
        start = time.time()
        with torch.no_grad():
            for sent in dev:
                # sentences = batch.to(DEVICE)
                loss = calc_sent_loss(sent, model, criterion).mean()
                dev_loss += loss.data
                dev_words += len(sent)

        # Keep track of the development accuracy and reduce the learning rate if it got worse
        if last_dev < dev_loss and type(optimizer) is not torch.optim.Adam:
            optimizer.learning_rate /= 2
        last_dev = dev_loss

        # Keep track of the best development accuracy, and save the model only if it's the best one
        if best_dev > dev_loss:
            if not os.path.exists('models'):
                try:
                    os.mkdir('models')
                except Exception as e:
                    print("Can not create models directory, %s" % e)
            torch.save(model.state_dict(), "models/model.pt")
            best_dev = dev_loss

        # Save the model
        logger.info("iter %r: dev loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (
            epoch, dev_loss / dev_words, math.exp(dev_loss / dev_words),
            dev_words / (time.time() - start)))

        # Generate a few sentences
        for _ in range(5):
            sentence = generate_sent(model)
            logger.debug(" ".join([word for word in sentence]))