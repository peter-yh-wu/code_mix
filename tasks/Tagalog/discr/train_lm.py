'''
Language Model Training Script

Peter Wu
peterw1@andrew.cmu.edu
'''

import argparse
import csv
import itertools
import numpy as np
import os
import sys
import time
import torch

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model import *
from model_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--save-directory', type=str, default='output/baseline/v1', help='output directory')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--num-workers', type=int, default=2, metavar='N', help='number of workers')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='N', help='lr')
    parser.add_argument('--weight-decay', type=float, default=1e-5, metavar='N', help='weight decay')
    parser.add_argument('--teacher-force-rate', type=float, default=0.9, metavar='N', help='teacher forcing rate')

    parser.add_argument('--word-dropout', type=float, default=0.2, metavar='N', help='word dropout')
    
    parser.add_argument('--emb-dim', type=int, default=300, metavar='N', help='hidden dimension')
    parser.add_argument('--hidden-dim', type=int, default=650, metavar='N', help='hidden dimension')

    return parser.parse_args()


def main():
    args = parse_args()

    t0 = time.time()

    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)
    LOG_PATH = os.path.join(args.save_directory, 'log')
    with open(LOG_PATH, 'w+') as ouf:
        pass

    print("Loading File IDs and Y Data")
    _, train_ys = load_fid_and_y_data('train')
    _, dev_ys = load_fid_and_y_data('dev')
    _, test_ys = load_fid_and_y_data('test')
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), LOG_PATH)

    print("Building Charset") # TODO this and below
    charset = build_charset(np.concatenate((train_ys, dev_ys, test_ys), axis=0))
    charmap = make_charmap(charset) # {string: int}
    charcount = len(charset)
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), LOG_PATH)

    print("Mapping Characters")
    trainchars = map_characters(train_ys, charmap) # list of 1-dim int np arrays
    devchars = map_characters(dev_ys, charmap) # list of 1-dim int np arrays
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), LOG_PATH)

    print("Building Loader")
    train_loader = make_loader(trainchars, args, shuffle=True, batch_size=args.batch_size)
    dev_loader = make_loader(devchars, args, shuffle=True, batch_size=args.batch_size)
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), LOG_PATH)

    print("Building Model")
    model = LSTMLM(charcount+1, args) # plus one for start/end token
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = SequenceCrossEntropy()
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), LOG_PATH)

    print("Running")
    ckpt_path = os.path.join(args.save_directory, 'model.ckpt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    if torch.cuda.is_available():
        model = model.cuda(args.cuda)

    best_val_loss = sys.maxsize
    prev_best_epoch = 0
    for e in range(args.epochs):
        t1 = time.time()
        print_log('Starting Epoch %d (%.2f Seconds)' % (e+1, t1-t0), LOG_PATH)

        # train
        model.train()
        optimizer.zero_grad()
        l = 0
        for i, t in enumerate(train_loader):
            l1array, llens, l2array = t # l1array shape: (maxlen, batch_size)
            l1array, llens, l2array = Variable(l1array), Variable(llens), Variable(l2array)
            if torch.cuda.is_available():
                l1array, llens, l2array = l1array.cuda(args.cuda), llens.cuda(args.cuda), l2array.cuda(args.cuda)
            logits, y_preds = model(l1array, llens)
            loss = criterion((logits, y_preds, llens), l2array)
            l += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            if (i+1) % 100 == 0:
                t1 = time.time()
                print('Processed %d Batches (%.2f Seconds)' % (i+1, t1-t0))
        print_log('Train Loss: %f' % (l/len(train_loader.dataset)), LOG_PATH)

        # val
        model.eval()
        with torch.no_grad():
            l = 0
            for i, t in enumerate(dev_loader):
                l1array, llens, l2array = t
                l1array, llens, l2array = Variable(l1array), Variable(llens), Variable(l2array)
                if torch.cuda.is_available():
                    l1array, llens, l2array = l1array.cuda(args.cuda), llens.cuda(args.cuda), l2array.cuda(args.cuda)
                logits, y_preds = model(l1array, llens)
                loss = criterion((logits, y_preds, llens), l2array)
                l += loss.item()
            val_loss = l/len(dev_loader.dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                prev_best_epoch = e
                torch.save(model.state_dict(), CKPT_PATH)
            elif e - prev_best_epoch > args.patience:
                break
            print_log('Val Loss: %f' % val_loss, LOG_PATH)


if __name__ == '__main__':
    main()
