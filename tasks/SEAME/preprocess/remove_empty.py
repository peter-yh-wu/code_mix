'''
Removes empty .mfcc files

Assumes that split_data.py has already been run

Peter Wu
peterw1@andrew.cmu.edu
'''

import itertools
import os
import numpy as np
import torch

from torch.autograd import Variable

def non_empty_ids(ids):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INTERVIEW_MFCC_DIR = os.path.join(parent_dir, 'data/interview/mfcc')
    mfcc_paths = [os.path.join(INTERVIEW_MFCC_DIR, fid+'.mfcc') for fid in ids]
    new_ids = []
    for i, path in enumerate(mfcc_paths):
        curr_mfcc = np.loadtxt(path) # shape: (seq_len, num_feats)
        if curr_mfcc.shape[0] > 0:
            new_ids.append(ids[i])
        if (i+1) % 500 == 0:
            print('removed %d files' % (i+1))
    return new_ids

TRAIN_IDS_FILE = 'train_ids.txt'
DEV_IDS_FILE = 'dev_ids.txt'
TEST_IDS_FILE = 'test_ids.txt'

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLIT_DIR = os.path.join(parent_dir, 'split')
train_ids_path = os.path.join(SPLIT_DIR, TRAIN_IDS_FILE)
dev_ids_path = os.path.join(SPLIT_DIR, DEV_IDS_FILE)
test_ids_path = os.path.join(SPLIT_DIR, TEST_IDS_FILE)

with open(train_ids_path, 'r') as inf:
    train_ids = inf.readlines()
train_ids = [f.strip() for f in train_ids]
with open(dev_ids_path, 'r') as inf:
    dev_ids = inf.readlines()
dev_ids = [f.strip() for f in dev_ids]
with open(test_ids_path, 'r') as inf:
    test_ids = inf.readlines()
test_ids = [f.strip() for f in test_ids]

print('removing empty train mfcc files')
train_ids = non_empty_ids(train_ids)
print('removing empty dev mfcc files')
dev_ids = non_empty_ids(dev_ids)
print('removing empty test mfcc files')
test_ids = non_empty_ids(test_ids)

with open(train_ids_path, 'w+') as ouf:
    for fid in train_ids:
        ouf.write('%s\n' % fid)
with open(dev_ids_path, 'w+') as ouf:
    for fid in dev_ids:
        ouf.write('%s\n' % fid)
with open(test_ids_path, 'w+') as ouf:
    for fid in test_ids:
        ouf.write('%s\n' % fid)