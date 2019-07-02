'''
Utils for Language Model Training Script

Peter Wu
peterw1@andrew.cmu.edu
'''

import itertools
import os
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class SequenceCrossEntropy(nn.CrossEntropyLoss):
    # Customized CrossEntropyLoss
    def __init__(self, *args, **kwargs):
        super(SequenceCrossEntropy, self).__init__(*args, reduce=False, **kwargs)

    def forward(self, prediction, target):
        logits, generated, sequence_lengths = prediction
        maxlen = logits.size(0)
        mask = Variable(output_mask(maxlen, sequence_lengths.data)).float()
        logits = logits * mask.unsqueeze(2)
        losses = super(SequenceCrossEntropy, self).forward(logits.view(-1, logits.size(2)), target.view(-1))
        loss = torch.sum(mask.view(-1) * losses) / logits.size(1)
        return loss


def output_mask(maxlen, lengths):
    '''Create a mask on-the-fly

    Args:
        maxlen: length of mask
        lengths: tensor with shape (batch_size,), comprised of
            length of each input sequence in batch
    
    Return:
        mask with shape (maxlen, batch_size)
    '''
    lens = lengths.unsqueeze(0) # shape (1, batch_size)
    range_tens = torch.arange(0, maxlen, 1, out=lengths.new()).unsqueeze(1)
    mask = range_tens < lens # shape: (maxlen, batch_size)
    return mask.transpose(0, 1)


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


class LMDataset(Dataset):
    def __init__(self, labels):
        '''
        Args:
            labels: list of 1-dim int np arrays
        '''
        self.labels = [torch.from_numpy(y + 1).long() for y in labels]  # +1 for start/end token
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.labels[index]


def text_collate_fn(batch):
    n = len(batch)
    llens = torch.IntTensor(n)
    for i, label in enumerate(batch):
        llens[i] = label.size(0) + 1 # +1 to account for start/end token
    lmax = int(llens.max())
    l1array = torch.LongTensor(lmax, n).zero_()
    l2array = torch.LongTensor(lmax, n).zero_()
    for i, label in enumerate(batch):
        l1array[1:label.size(0) + 1, i] = label
        l2array[:label.size(0), i] = label
    return l1array, llens, l2array 


def make_loader(labels, args, shuffle=True, batch_size=64):
    '''
    Args:
        features: list of file id strings (files contain x values)
        labels: list of 1-dim int np arrays
    '''
    # Build the DataLoaders
    kwargs = {'pin_memory': True, 'num_workers': args.num_workers} if torch.cuda.is_available() else {}
    dataset = LMDataset(labels)
    loader = DataLoader(dataset, collate_fn=text_collate_fn, shuffle=shuffle, batch_size=batch_size, **kwargs)
    return loader