'''
Helper functions

To-do:
 - incrementally load data instead all at once
 - add conversation data to split

Character set/map and data loader code modified from LAS implementation by 
Sai Krishna Rallabandi (srallaba@andrew.cmu.edu)

Peter Wu
peterw1@andrew.cmu.edu
'''

import itertools
import os
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

def print_log(s, log_path):
    print(s)
    with open(log_path, 'a+') as ouf:
        ouf.write("%s\n" % s)

def build_charset(utterances):
    # Create a character set
    chars = set(itertools.chain.from_iterable(utterances))
    chars = list(chars)
    chars.sort()
    return chars

def make_charmap(charset):
    # Create the inverse character map
    return {c: i for i, c in enumerate(charset)}

def map_characters(utterances, charmap):
    # Convert transcripts to ints
    ints = [np.array([charmap[c] for c in u], np.int32) for u in utterances]
    return ints

def load_ids():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SPLIT_DIR = os.path.join(parent_dir, 'split')
    TRAIN_IDS_FILE = 'train_ids.txt'
    DEV_IDS_FILE = 'dev_ids.txt'
    TEST_IDS_FILE = 'test_ids.txt'
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
    return train_ids, dev_ids, test_ids

def load_x_data(ids):
    '''Returns list comprised of shape(seq_len, num_feats) np arrays'''
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INTERVIEW_MFCC_DIR = os.path.join(parent_dir, 'data/interview/mfcc')
    mfcc_paths = [os.path.join(INTERVIEW_MFCC_DIR, fid+'.mfcc') for fid in ids]
    mfccs = []
    for path in mfcc_paths:
        curr_mfcc = np.loadtxt(path) # shape: (seq_len, num_feats)
        mfccs.append(curr_mfcc)
    return mfccs

def load_y_data():
    '''Returns 3 1-dim np arrays of strings'''
    SPLIT_DIR = os.path.join(parent_dir, 'split')
    TRAIN_YS_FILE = 'train_ys.txt'
    DEV_YS_FILE = 'dev_ys.txt'
    TEST_YS_FILE = 'test_ys.txt'
    train_ys_path = os.path.join(SPLIT_DIR, TRAIN_YS_FILE)
    dev_ys_path = os.path.join(SPLIT_DIR, DEV_YS_FILE)
    test_ys_path = os.path.join(SPLIT_DIR, TEST_YS_FILE)
    with open(train_ys_path, 'r') as inf:
        train_ys = inf.readlines()
    train_ys = [f.strip() for f in train_ys]
    with open(dev_ys_path, 'r') as inf:
        dev_ys = inf.readlines()
    dev_ys = [f.strip() for f in dev_ys]
    with open(test_ys_path, 'r') as inf:
        test_ys = inf.readlines()
    test_ys = [f.strip() for f in test_ys]
    return np.array(train_ys), np.array(dev_ys), np.array(test_ys)

class SpeechDataset(Dataset):
    '''Assumes all characters in transcripts are alphanumeric'''
    def __init__(self, features, transcripts):
        '''
        Args:
            features: list of shape(seq_len, num_feats) np arrays
            transcripts: list of 1-dim int np arrays
        '''
        self.features = [torch.from_numpy(x).float() for x in features]
        if transcripts:
            self.transcripts = [torch.from_numpy(y + 1).long() for y in transcripts]  # +1 for start/end token
            assert len(self.features) == len(self.transcripts)
        else:
            self.transcripts = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        if self.transcripts:
            return self.features[item], self.transcripts[item]
        else:
            return self.features[item], None

INPUT_DIM = 66

def speech_collate_fn(batch):
    n = len(batch)

    # allocate tensors for lengths
    ulens = torch.IntTensor(n)
    llens = torch.IntTensor(n)

    # calculate lengths
    for i, (u, l) in enumerate(batch):
        # +1 to account for start/end token
        ulens[i] = u.size(0)
        if l is None:
            llens[i] = 1
        else:
            llens[i] = l.size(0) + 1

    # calculate max length
    umax = int(ulens.max())
    lmax = int(llens.max())

    # allocate tensors for data based on max length
    uarray = torch.FloatTensor(umax, n, INPUT_DIM).zero_()
    l1array = torch.LongTensor(lmax, n).zero_()
    l2array = torch.LongTensor(lmax, n).zero_()

    # collate data tensors into pre-allocated arrays
    for i, (u, l) in enumerate(batch):
        uarray[:u.size(0), i, :] = u
        if l is not None:
            l1array[1:l.size(0) + 1, i] = l
            l2array[:l.size(0), i] = l

    return uarray, ulens, l1array, llens, l2array

def make_loader(features, labels, args, shuffle=True, batch_size=64):
    '''
    Args:
        features: len-num_samples list
        labels: list of 1-dim int np arrays
    '''
    # Build the DataLoaders
    kwargs = {'pin_memory': True, 'num_workers': args.num_workers} if args.cuda else {}
    dataset = SpeechDataset(features, labels)
    loader = DataLoader(dataset, collate_fn=speech_collate_fn, shuffle=shuffle, batch_size=batch_size, **kwargs)
    return loader