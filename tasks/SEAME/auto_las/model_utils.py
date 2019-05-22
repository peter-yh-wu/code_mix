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

        logits, generated, lens = model(
            uarray, ulens, l1array, llens,
            future=args.generator_length)
        generated = generated.data.cpu().numpy()  # (L, BS)
        n = uarray.size(1)
        for i in range(n):
            transcript = decode_output(generated[:, i], charset)
            yield transcript

def cer(args, model, loader, charset, ys, truncate=True):
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
        if truncate:
            dist = edit_distance(t[:len(ys[i])], ys[i])
        else:
            dist = edit_distance(t, ys[i])
        norm_dist = dist / len(ys[i])
        norm_dists.append(norm_dist)
    return sum(norm_dists)/len(ys)

def cer_from_transcripts(transcripts, ys, log_path, truncate=True, spaces='best'):
    '''
    Return:
        norm_dists: list of CER values
        dist: edit distances
        spaces: no, yes, best (to account for incongruity in raw data spacing)
    '''
    norm_dists = []
    dists = []
    for i, t in enumerate(transcripts):
        curr_t = t
        curr_y = ys[i]
        if len(curr_y) == 0:
            print('%d is 0' % i)
        curr_t_nos = curr_t.replace(' ', '')
        curr_y_nos = curr_y.replace(' ', '')
        if truncate:
            curr_t = curr_t[:len(curr_y)]
            curr_t_nos = curr_t_nos[:len(curr_y_nos)]
        dist = edit_distance(curr_t, curr_y)
        norm_dist = dist / len(curr_y)
        dist_nos = edit_distance(curr_t_nos, curr_y_nos)
        norm_dist_nos = dist_nos / len(curr_y_nos)
        best_dist = min(dist, dist_nos)
        best_norm = min(norm_dist, norm_dist_nos)
        with open(log_path, 'a') as ouf:
            ouf.write('dist: %.2f, norm_dist: %.2f\n' % (best_dist, best_norm))
        norm_dists.append(best_norm)
        dists.append(best_dist)
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
    def __init__(self, paths, labels=None, lids=None):
        '''
        self.labels is only True for test set

        Args:
            ids: list of file id strings (files contain x values)
            labels: list of 1-dim int np arrays
            lids: list of 1-dim int np arrays
        '''
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.paths = paths
        if labels:
            self.labels = [torch.from_numpy(y + 1).long() for y in labels]  # +1 for start token
            # assert len(self.paths) == len(self.labels)
        else:
            self.labels = None
        if lids:
            self.lids = [torch.from_numpy(np.array(y)).long() for y in lids]
            assert len(self.paths) == len(self.lids)
        else:
            self.lids = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        curr_path = self.paths[index]
        curr_mfcc = torch.from_numpy(np.loadtxt(curr_path)).float()
        
        curr_label = None if not self.labels else self.labels[index]
        curr_lid = None if not self.lids else self.lids[index]
        # assert torch.max(curr_lid) <= 1

        return curr_mfcc, curr_label, curr_lid

INPUT_DIM = 39

def speech_collate_fn(batch):
    n = len(batch)

    # allocate tensors for lengths
    ulens = torch.IntTensor(n)
    ylens = torch.IntTensor(n)
    lid_lens = torch.IntTensor(n)

    # calculate lengths
    for i, (u, y, lid) in enumerate(batch): # u is x-val, y is y-val
        # assert torch.max(lid) <= 1
        # +1 to account for start/end token
        ulens[i] = u.size(0)
        if y is None:
            ylens[i] = 1
        else:
            ylens[i] = y.size(0) + 1
        if lid is None:
            lid_lens[i] = 1
        else:
            lid_lens[i] = lid.size(0) + 1

    # calculate max length
    umax = int(ulens.max())
    lmax = int(ylens.max())
    lid_max = int(lid_lens.max())
    # assert(lmax == lid_max)

    # allocate tensors for data based on max length
    uarray = torch.FloatTensor(umax, n, INPUT_DIM).zero_()
    y1array = torch.LongTensor(lmax, n).zero_()
    y2array = torch.LongTensor(lmax, n).zero_()
    lid1_arr = torch.LongTensor(lmax, n).zero_()
    lid2_arr = torch.LongTensor(lmax, n).zero_()

    # collate data tensors into pre-allocated arrays
    for i, (u, y, lid) in enumerate(batch):
        uarray[:u.size(0), i, :] = u
        if y is not None:
            y1array[1:y.size(0) + 1, i] = y
            y2array[:y.size(0), i] = y
        if lid is not None:
            lid1_arr[1:lid.size(0) + 1, i] = lid
            lid2_arr[:lid.size(0), i] = lid

    return uarray, ulens, y1array, ylens, y2array, lid1_arr, lid2_arr

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
    with open(ys_path, 'r', encoding="utf-8") as inf:
        ys = inf.readlines()
    ys = [y.strip() for y in ys]
    return np.array(ys)

def load_lids(stage):
    '''return 2d list of ints'''
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SPLIT_DIR = os.path.join(parent_dir, 'split')
    FILE = '%s_lids.txt' % stage
    ys_path = os.path.join(SPLIT_DIR, FILE)
    with open(ys_path, 'r') as inf:
        ys = inf.readlines()
    ys = [y.strip() for y in ys]
    ys = [[int(lid) for lid in y] for y in ys]

    ys_flat = [item for sublist in ys for item in sublist]
    # assert max(ys_flat) <= 1
    
    return ys

def load_switch_lids(stage):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SPLIT_DIR = os.path.join(parent_dir, 'split')
    FILE = '%s_switch_lids.txt' % stage
    ys_path = os.path.join(SPLIT_DIR, FILE)
    with open(ys_path, 'r') as inf:
        ys = inf.readlines()
    ys = [y.strip() for y in ys]
    ys = [y.split() for y in ys]
    ys = [[int(lid) for lid in y] for y in ys]
    return ys

def make_loader(ids, labels, args, lids=None, shuffle=True, batch_size=64):
    '''
    Args:
        features: list of file id strings (files contain x values)
        labels: list of 1-dim int np arrays
    '''
    # Build the DataLoaders
    kwargs = {'pin_memory': True, 'num_workers': args.num_workers} if args.cuda else {}
    dataset = ASRDataset(ids, labels=labels, lids=lids)
    loader = DataLoader(dataset, collate_fn=speech_collate_fn, shuffle=shuffle, batch_size=batch_size, **kwargs)
    return loader