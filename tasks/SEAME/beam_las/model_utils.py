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

from nltk.metrics import edit_distance
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

def output_mask(maxlen, lengths):
    """
    Create a mask on-the-fly
    :param maxlen: length of mask
    :param lengths: length of each sequence
    :return: mask shaped (maxlen, len(lengths))
    """
    lens = lengths.unsqueeze(0)
    ran = torch.arange(0, maxlen, 1, out=lengths.new()).unsqueeze(1)
    mask = ran < lens
    return mask

def log_l(logits, target, lengths):
    '''Calculates the log-likelihood for the given batch

    Args:
        logits: shape (seq_len, batch_size, vocab_size)
        target: shape (seq_len, batch_size)
        lengths: shape (batch_size,)
    
    Return:
        log_probs: shape (batch_size,)
    '''
    seq_len, batch_size, vocab_size = logits.shape
    mask = output_mask(seq_len, lengths.data).float()
    logits_masked = logits * mask.unsqueeze(2)
    range_tens = torch.arange(vocab_size).repeat(seq_len, batch_size, 1)
    if torch.cuda.is_available():
        range_tens = range_tens.cuda()
    target_rep = target.repeat(vocab_size, 1, 1).permute(1, 2, 0)
    masked_tens = range_tens == target_rep
    all_probs = torch.sum(logits*masked_tens.float(), 2) # shape: (seq_len, batch_size)
    all_probs = all_probs.clamp(min=1e-20)
    all_log_probs = torch.log(all_probs)
    log_probs = torch.sum(all_log_probs, 0) # shape: (batch_size,)
    return log_probs

def perplexities_from_x(model, loader):
    '''
    Return:
        np array of floats with same number of elements as len(loader)
    '''
    model.eval()

    all_perps = np.array([])
    for uarray, ulens, l1array, llens, l2array in loader:
        uarray, ulens, l1array, llens, l2array = Variable(uarray), \
            Variable(ulens), Variable(l1array), Variable(llens), Variable(l2array)
        if torch.cuda.is_available():
            uarray, ulens, l1array, llens, l2array = uarray.cuda(), \
                ulens.cuda(), l1array.cuda(), llens.cuda(), l2array.cuda()
        prediction = model(uarray, ulens, l1array, llens)
        logits, generated, char_lengths = prediction
        perps = perplexities(logits, l2array, char_lengths) # shape: (batch_size,)
        perps_np = perps.cpu.numpy()
        all_perps = np.append(all_perps, perps_np)
    return all_perps

def perplexity(logits, target, lengths):
    '''Calculates the perplexity for the given batch

    Args:
        logits: shape (seq_len, batch_size, vocab_size)
        target: shape (seq_len, batch_size)
        lengths: shape (batch_size,)
    
    Return:
        perp: float (tensor)
    '''
    log_probs = log_l(logits, target, lengths) # shape: (batch_size,)
    tot_log_l = torch.sum(log_probs)
    tot_len = torch.sum(lengths)
    return torch.exp(-tot_log_l/tot_len)

def perplexities(logits, target, lengths):
    log_probs = log_l(logits, target, lengths) # shape: (batch_size,)
    return torch.exp(-log_probs/lengths) # shape: (batch_size,)

def decode_output(output, charset):
    # Convert ints back to strings
    chars = []
    for o in output:
        if o == 0:
            break
        chars.append(charset[o - 1])
    return "".join(chars)

def generate_transcripts(args, model, loader, charset):
    '''Iteratively returns string transcriptions
    
    Return:
        generator object comprised of transcripts (each a string)
    '''
    # Create and yield transcripts
    for uarray, ulens, l1array, llens, l2array in loader:
        if args.cuda:
            uarray = uarray.cuda()
            ulens = ulens.cuda()
            l1array = l1array.cuda()
            llens = llens.cuda()
        uarray = Variable(uarray)
        ulens = Variable(ulens)
        l1array = Variable(l1array)
        llens = Variable(llens)

        logits, generated, lens = model(uarray, ulens, l1array, llens)
        generated = generated.data.cpu().numpy()  # (L, BS)
        n = uarray.size(1)
        for i in range(n):
            transcript = decode_output(generated[:, i], charset)
            yield transcript

def cer(args, model, loader, charset, ys):
    '''Calculates the average normalized CER for the given data
    
    Args:
        ys: iterable of strings
    
    Return:
        number
    '''
    model.eval()
    norm_dists = []
    transcripts = generate_transcripts(args, model, loader, charset)
    for i, t in enumerate(transcripts):
        dist = edit_distance(t, ys[i])
        norm_dist = dist / len(ys[i])
        norm_dists.append(norm_dist)
    return sum(norm_dists)/len(ys)

def cer_from_transcripts(transcripts, ys, log_path, truncate=True):
    '''
    Return:
        norm_dists: list of CER values
        dist: edit distances
    '''
    norm_dists = []
    dists = []
    empty_ys = []
    for i, t in enumerate(transcripts):
        if len(ys[i]) < 1:
            empty_ys.append(i)
            continue
        if truncate:
            dist = edit_distance(t[:len(ys[i])], ys[i])
        else:
            dist = edit_distance(t, ys[i])
        norm_dist = dist / len(ys[i])
        with open(log_path, 'a') as ouf:
            ouf.write('dist: %.2f, norm_dist: %.2f\n' % (dist, norm_dist))
        norm_dists.append(norm_dist)
        dists.append(dist)

    print('Empty targets', empty_ys)
    return norm_dists, dists

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

def load_paths():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SPLIT_DIR = os.path.join(parent_dir, 'split')
    TRAIN_PATHS_FILE = 'train_paths.txt'
    DEV_PATHS_FILE = 'dev_paths.txt'
    TEST_PATHS_FILE = 'test_paths.txt'
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
    return train_paths, dev_paths, test_paths

class ASRDataset(Dataset):
    '''Assumes all characters in transcripts are alphanumeric'''
    def __init__(self, paths, labels=None):
        '''
        self.labels is only True for test set

        Args:
            ids: list of file id strings (files contain x values)
            labels: list of 1-dim int np arrays
        '''
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.paths = paths
        if labels:
            self.labels = [torch.from_numpy(y + 1).long() for y in labels]  # +1 for start/end token
            assert len(self.paths) == len(self.labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        curr_path = self.paths[index]
        curr_mfcc = torch.from_numpy(np.loadtxt(curr_path)).float()
        
        if self.labels:
            return curr_mfcc, self.labels[index]
        else:
            return curr_mfcc, None

INPUT_DIM = 39

def speech_collate_fn(batch):
    n = len(batch)

    # allocate tensors for lengths
    ulens = torch.IntTensor(n)
    llens = torch.IntTensor(n)

    # calculate lengths
    for i, (u, l) in enumerate(batch): # u is x-val, l is y-val
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

def load_y_data(stage):
    '''
    Assumes that y-values are aligned with file ids (specified by indices)

    Args:
        stage: train, dev, or test
    
    Return:
        1-dim np array of strings
    '''
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SPLIT_DIR = os.path.join(parent_dir, 'split')
    FILE = '%s_ys.txt' % stage
    ys_path = os.path.join(SPLIT_DIR, FILE)
    with open(ys_path, 'r') as inf:
        ys = inf.readlines()
    ys = [y.strip() for y in ys]
    return np.array(ys)

def make_loader(ids, labels, args, shuffle=True, batch_size=64):
    '''
    Args:
        features: list of file id strings (files contain x values)
        labels: list of 1-dim int np arrays
    '''
    # Build the DataLoaders
    kwargs = {'pin_memory': True, 'num_workers': args.num_workers} if args.cuda else {}
    dataset = ASRDataset(ids, labels)
    loader = DataLoader(dataset, collate_fn=speech_collate_fn, shuffle=shuffle, batch_size=batch_size, **kwargs)
    return loader