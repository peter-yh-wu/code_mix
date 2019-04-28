'''
Script to evaluate model

Assumes that model.ckpt exists
Supported test-mode values: transcript, cer, perp, and all combos

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

from baseline import parse_args, Seq2SeqModel, write_transcripts
from model_utils import *

def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    t0 = time.time()

    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)
    LOG_PATH = os.path.join(args.save_directory, 'log')
    with open(LOG_PATH, 'w+') as ouf:
        pass

    print("Loading File Paths")
    train_paths, dev_paths, test_paths = load_paths()
    train_paths, dev_paths, test_paths = train_paths[:args.max_train], dev_paths[:args.max_dev], test_paths[:args.max_test]
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), LOG_PATH)

    print("Loading Y Data")
    test_paths = test_paths[:args.max_data]
    train_ys = load_y_data('train') # 1-dim np array of strings
    dev_ys = load_y_data('dev')
    test_ys = load_y_data('test')
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), LOG_PATH)

    print("Building Charset")
    charset = build_charset(np.concatenate((train_ys, dev_ys, test_ys), axis=0))
    charmap = make_charmap(charset) # {string: int}
    charcount = len(charset)
    t1 = time.time()
    print_log('%.2f Seconds' % (t1-t0), LOG_PATH)

    print("Mapping Characters")
    testchars = map_characters(test_ys, charmap)
    print("Building Loader")
    test_loader = make_loader(test_paths, testchars, args, shuffle=False, batch_size=1)

    print("Building Model")
    model = Seq2SeqModel(args, vocab_size=charcount, beam_width=args.beam_width)

    CKPT_PATH = os.path.join(args.save_directory, 'model.ckpt')
    if args.cuda:
        model.load_state_dict(torch.load(CKPT_PATH))
    else:
        gpu_dict = torch.load(CKPT_PATH, map_location=lambda storage, loc: storage)
        cpu_model_dict = {}
        for key, val in gpu_dict.items():
            cpu_model_dict[key] = val.cpu()
        model.load_state_dict(cpu_model_dict)
    print("Loaded Checkpoint")

    if args.cuda:
        model = model.cuda()

    model.eval()

    TRANSCRIPT_LOG_PATH = os.path.join(args.save_directory, 'transcript_log.txt')
    CSV_PATH = os.path.join(args.save_directory, 'submission.csv')

    if 'transcript' in args.test_mode:
        print('generating transcripts')
        with open(TRANSCRIPT_LOG_PATH, 'w+') as ouf:
            pass
        if not os.path.exists(CSV_PATH):
            transcripts = write_transcripts(
                path=CSV_PATH,
                args=args, model=model, loader=test_loader, charset=charset,
                log_path=TRANSCRIPT_LOG_PATH
            )
        else:
            transcripts = []
            with open(CSV_PATH, 'r') as csvfile:
                raw_csv = csv.reader(csvfile)
                for row in raw_csv:
                    with open(TRANSCRIPT_LOG_PATH, 'a') as ouf:
                        ouf.write('%s\n' % row[1])
                    transcripts.append(row[1])
        t1 = time.time()
        print("Finshed Writing Transcripts")
        print('%.2f Seconds' % (t1-t0))

    if 'cer' in args.test_mode:
        print('calculating cer values')
        CER_LOG_PATH = os.path.join(args.save_directory, 'cer_log.txt')
        with open(CER_LOG_PATH, 'w+') as ouf:
            pass
        transcripts = []
        with open(CSV_PATH, 'r') as csvfile:
                raw_csv = csv.reader(csvfile)
                for row in raw_csv:
                    transcripts.append(row[1])
        transcripts = [l.strip() for l in transcripts]
        CER_PATH = os.path.join(args.save_directory, 'test_cer.npy')
        EDIT_PATH = os.path.join(args.save_directory, 'test_edit.npy')
        norm_dists, dists = cer_from_transcripts(transcripts, test_ys, CER_LOG_PATH)
        np.save(CER_PATH, norm_dists)
        np.save(EDIT_PATH, dists)

    if 'perp' in args.test_mode:
        print('calculating perp values')
        PERP_LOG_PATH = os.path.join(args.save_directory, 'perp_log.txt')
        with open(PERP_LOG_PATH, 'w+') as ouf:
            pass
        PERP_PATH = os.path.join(args.save_directory, 'test_perp.npy')
        all_perps = perplexities_from_x(model, test_loader)
        np.save(PERP_PATH, all_perps)

if __name__ == '__main__':
    main()
