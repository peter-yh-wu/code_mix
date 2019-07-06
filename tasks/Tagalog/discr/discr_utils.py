'''
Utils for Discriminator Training Script

Peter Wu
peterw1@andrew.cmu.edu
'''

import itertools
import numpy as np
import os
import pickle
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset


def print_log(s, log_path):
    print(s)
    with open(log_path, 'a+') as ouf:
        ouf.write("%s\n" % s)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_fid_and_y_data(phase):
    '''
    Return:
        ids: list of file ids
        ys: 1-dim np array of strings
    '''
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_dir = os.path.join(parent_dir, 'split')
    phase_file = '%s.txt' % phase
    phase_path = os.path.join(split_dir, phase_file)
    with open(phase_path, 'r') as inf:
        lines = inf.readlines()
    ids = []
    ys = []
    for l in lines:
        l = l.strip()
        l_list = l.split()
        fid = l_list[0]
        y = ' '.join(l_list[1:])
        ids.append(fid)
        ys.append(y)
    return ids, np.array(ys)


def load_gens():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    gens_path = os.path.join(data_dir, 'discr', 'gs.pkl')
    return load_pkl(gens_path)

def mk_fid_to_orig(fids, ys):
    fid_to_orig = {}
    for fid, y in zip(fids, ys):
        fid_to_orig[fid] = y
    return fid_to_orig


def build_charset(utterances):
    # Create a character set
    chars = set(itertools.chain.from_iterable(utterances))
    chars = list(chars)
    chars.sort()
    return chars


def make_charmap(charset):
    # Create the inverse character map
    return {c: i for i, c in enumerate(charset)}


def simplify_gens(fid_to_gens, fid_to_orig):
    new_fid_to_gens = {}
    for fid in fid_to_gens:
        gens = fid_to_gens[fid]
        if fid not in fid_to_orig:
            new_gens = gens
        else:
            new_gens = []
            for g in gens:
                if g != fid_to_orig[fid]:
                    new_gens.append(g)
        new_fid_to_gens[fid] = new_gens
    return new_fid_to_gens


def map_characters_orig(fid_to_orig, charmap):
    '''Convert transcripts to ints
    '''
    new_fid_to_orig = {fid: np.array([charmap[c] for c in u], np.int32) for (fid, u) in fid_to_orig.items()}
    return new_fid_to_orig


def map_characters_gens(fid_to_gens, charmap):
    '''Convert transcripts to ints
    '''
    new_fid_to_gens = {}
    for fid in fid_to_gens:
        gens = fid_to_gens[fid]
        new_gens = []
        for g in gens:
            new_g = np.array([charmap[c] for c in g], np.int32)
            new_gens.append(new_g)
        new_fid_to_gens[fid] = new_gens
    return new_fid_to_gens


class SimpleDiscrDataset(Dataset):
    def __init__(self, fid_to_orig, fid_to_gens):
        '''
        Args:
            fid_to_orig: {fid string: orig np array of ints}
            fid_to_gens: {fid string: [g1 np array of ints, g2 np array of ints, ...]}
        '''
        self.fids = list(fid_to_orig.keys())
        self.fid_to_orig = fid_to_orig
        self.fid_to_gens = fid_to_gens

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, index):
        fid = self.fids[index]
        if fid not in self.fid_to_gens:
            return self.fid_to_orig[fid], []
        else:
            return self.fid_to_orig[fid], self.fid_to_gens[fid]


def simple_discr_collate_fn(batch):
    '''
    Args:
        batch: list of (orig, gens) pairs, where orig is np array of ints and
            gens is list of np array of ints
    
    Return:
        xs: LongTensor with shape (batch_size, max_len)
        ys: LongTensor with shape (batch_size,)
            real (orig) is 1, fake (gen) is 0
    '''
    batch_size = len(batch)
    max_len = 0
    for (orig, gens) in batch:
        batch_size += len(gens)
        max_len = max([max_len, len(orig)]+[len(g) for g in gens])
    xs = torch.LongTensor(batch_size, max_len).zero_()
    ys = torch.LongTensor(batch_size).zero_()
    i = 0
    for (orig, gens) in batch:
        xs[i, :len(orig)] = torch.from_numpy(orig).long()
        ys[i] = 1
        for g in gens:
            i += 1
            xs[i, :len(g)] = torch.from_numpy(g).long()
            ys[i] = 0
    return xs, ys


def make_simple_loader(fid_to_orig, fid_to_gens, args, shuffle=True, batch_size=64):
    kwargs = {'pin_memory': True, 'num_workers': args.num_workers} if torch.cuda.is_available() else {}
    dataset = SimpleDiscrDataset(fid_to_orig, fid_to_gens)
    loader = DataLoader(dataset, collate_fn=simple_discr_collate_fn, shuffle=shuffle, batch_size=batch_size, **kwargs)
    return loader