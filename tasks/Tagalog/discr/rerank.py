'''
python3 rerank.py --beam-file beams.csv # plus other args used during training

Assumes that beam outputs file is in args.save_directory
If args.beam_file is not provided, assumes that beam outputs file is preds.csv

e.g.
python3 rerank.py --save-directory output/lstm3 --beam-file submission_beam_10.csv --emb-dim 300 --hidden-dim 650 --num-layers 3
'''

import os

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

from discr_utils import *
from model import *
from train_simple_discr import parse_args


def find_best_pred(model, preds):
    '''
    Args:
        model: nn.Module, discriminator
        preds: list of 1-dim np int arrays
    '''
    best_prob_real = -sys.maxsize-1
    best_i = 0
    for i, p in enumerate(preds):
        logits = model(p.unsqueeze(0))
        prob_real = logits.cpu()[0][1].item()
        if prob_real > best_prob_real:
            best_i = i
            best_prob_real = prob_real
    return preds[best_i], best_i


def main():
    args = parse_args()

    log_path = os.path.join(args.save_directory, 'rerank_log')
    t0 = time.time()

    print("Loading File IDs and Y Data")
    train_fids, train_orig = load_fid_and_y_data('train')
    dev_fids, dev_orig = load_fid_and_y_data('dev')
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), log_path)

    print("Building Charset")
    charset = build_charset(np.concatenate((train_orig, dev_orig), axis=0))
    charmap = make_charmap(charset) # {string: int}
    charcount = len(charset)
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), log_path)

    print("Building Model")
    model = SimpleLSTMDiscriminator(charcount, num_layers=args.num_layers, word_dropout=args.word_dropout, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), log_path)
    
    print("Running")
    ckpt_path = os.path.join(args.save_directory, 'discr.ckpt')
    if os.path.exists(ckpt_path):
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(ckpt_path))
        else:
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    if torch.cuda.is_available():
        model = model.cuda(args.cuda)

    print("Mapping Characters")
    preds_path = os.path.join(args.save_directory, args.beam_file)
    raw_preds = load_preds(preds_path) # list of string lists
    all_preds = map_characters_rerank(raw_preds, charmap)
        # list of lists, each sublist comprised of 1-dim int np arrays
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), log_path)

    print("Reranking")
    reranked_preds = []
    model.eval()
    with torch.no_grad():
        for pred_i, preds in enumerate(all_preds):
            best_pred, best_i = find_best_pred(model, preds)
            best_pred_str = raw_preds[pred_i][best_i]
            reranked_preds.append(best_pred_str)
            if (pred_i+1) % 100 == 0:
                t1 = time.time()
                print('Processed %d / %d groups (%.2f Seconds)' % (pred_i+1, len(all_preds), t1-t0))
    reranked_preds_path = os.path.join(args.save_directory, 'reranked.csv')
    with open(reranked_preds_path, 'w+', newline='') as f:
        w = csv.writer(f)
        for i, t in enumerate(reranked_preds):
            w.writerow([i+1, t])
            with open(log_path, 'a') as ouf:
                ouf.write('%s\n' % t)

if __name__ == "__main__":
    main()