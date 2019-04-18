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

def non_empty_paths(paths):
    new_paths = []
    for i, path in enumerate(paths):
        curr_mfcc = np.loadtxt(path) # shape: (seq_len, num_feats)
        if curr_mfcc.shape[0] > 0:
            new_paths.append(path)
        if (i+1) % 500 == 0:
            print('checked %d files' % (i+1))
    return new_paths

TRAIN_PATHS_FILE = 'train_paths.txt'
DEV_PATHS_FILE = 'dev_paths.txt'
TEST_PATHS_FILE = 'test_paths.txt'

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(parent_dir, 'split')
train_paths_path = os.path.join(SPLIT_DIR, TRAIN_PATHS_FILE)
dev_paths_path = os.path.join(SPLIT_DIR, DEV_PATHS_FILE)
test_paths_path = os.path.join(SPLIT_DIR, TEST_PATHS_FILE)

with open(train_paths_path, 'r') as inf:
    train_paths = inf.readlines()
train_paths = [f.strip() for f in train_paths]
with open(dev_paths_path, 'r') as inf:
    dev_paths = inf.readlines()
dev_paths = [f.strip() for f in dev_paths]
with open(test_paths_path, 'r') as inf:
    test_paths = inf.readlines()
test_paths = [f.strip() for f in test_paths]

print('removing empty train mfcc files')
train_paths = non_empty_paths(train_paths)
print('removing empty dev mfcc files')
dev_paths = non_empty_paths(dev_paths)
print('removing empty test mfcc files')
test_paths = non_empty_paths(test_paths)

with open(train_paths_path, 'w+') as ouf:
    for path in train_paths:
        ouf.write('%s\n' % path)
with open(dev_paths_path, 'w+') as ouf:
    for path in dev_paths:
        ouf.write('%s\n' % path)
with open(test_paths_path, 'w+') as ouf:
    for path in test_paths:
        ouf.write('%s\n' % path)