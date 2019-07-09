'''
Helper functions

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

def log_l(logits, target, lengths, device=0):
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
        range_tens = range_tens.cuda(device)
    target_rep = target.repeat(vocab_size, 1, 1).permute(1, 2, 0)
    masked_tens = range_tens == target_rep
    all_probs = torch.sum(logits*masked_tens.float(), 2) # shape: (seq_len, batch_size)
    all_probs = all_probs.clamp(min=1e-10)
    all_log_probs = torch.log(all_probs)
    log_probs = torch.sum(all_log_probs, 0) # shape: (batch_size,)
    return log_probs

def perplexity(logits, target, lengths, device=0):
    '''Calculates the perplexity for the given batch

    Args:
        logits: shape (seq_len, batch_size, vocab_size)
        target: shape (seq_len, batch_size)
        lengths: shape (batch_size,)
    
    Return:
        perp: float (tensor)
    '''
    log_probs = log_l(logits, target, lengths, device=device) # shape: (batch_size,)
    tot_log_l = torch.sum(log_probs)
    tot_len = torch.sum(lengths)
    return torch.exp(-tot_log_l/tot_len)

def decode_output(output, charset):
    # Convert ints back to strings
    chars = []
    for o in output:
        if o == 0:
            break
        chars.append(charset[o - 1])
    return "".join(chars)

def generate_transcripts(args, model, loader, charset, device=0):
    '''Iteratively returns string transcriptions
    
    Return:
        generator object comprised of transcripts (each a string)
    '''
    # Create and yield transcripts
    for uarray, ulens, l1array, llens, l2array in loader:
        if torch.cuda.is_available():
            uarray = uarray.cuda(device)
            ulens = ulens.cuda(device)
            l1array = l1array.cuda(device)
            llens = llens.cuda(device)
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

def cer(args, model, loader, charset, ys, device=0, truncate=True):
    '''Calculates the average normalized CER for the given data
    
    Args:
        ys: iterable of strings
    
    Return:
        number
    '''
    model.eval()
    norm_dists = []
    transcripts = generate_transcripts(args, model, loader, charset, device=device)
    for i, t in enumerate(transcripts):
        if truncate:
            dist = edit_distance(t[:len(ys[i])], ys[i])
        else:
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

def load_fid_and_y_data(phase):
    '''Loads .mfcc file ids and y values for given phase

    Args:
        phase: 'train', 'dev', or 'test'

    Return:
        ids: list of file ids
        ys: 1-dim np array of strings; ys[i] is transcription of .mfcc file
            with id ids[i]
    '''
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_dir = os.path.join(parent_dir, 'split')
    phase_file = '%s.txt' % phase
    phase_path = os.path.join(split_dir, phase_file)
    with open(phase_path, 'r', encoding="utf-8") as inf:
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

def load_fid_and_y_data_miami(phase):
    '''Loads .mfcc file ids and y values for given phase

    Args:
        phase: 'train', 'dev', or 'test'

    Return:
        ids: list of file ids
        ys: 1-dim np array of strings; ys[i] is transcription of .mfcc file
            with id ids[i]
    '''
    gparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    split_dir = os.path.join(gparent_dir, 'Miami', 'split')
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

class ASRDataset(Dataset):
    '''Assumes all characters in transcripts are alphanumeric'''
    def __init__(self, ids, labels=None):
        '''
        self.labels is only True for test set

        Args:
            ids: list of file id strings (files contain x values)
            labels: list of 1-dim int np arrays
        '''
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.mfcc_dir = os.path.join(parent_dir, 'data/mfcc')
        mfcc_files = os.listdir(self.mfcc_dir)
        mfcc_paths_set = set([os.path.join(self.mfcc_dir, f) for f in mfcc_files])
        self.ids = ids
        if labels:
            self.labels = [torch.from_numpy(y + 1).long() for y in labels]  # +1 for start/end token
            new_ids = []
            new_labels = []
            for i, label in enumerate(self.labels):
                curr_id = self.ids[i]
                curr_mfcc_path = os.path.join(self.mfcc_dir, curr_id+'.mfcc')
                if curr_mfcc_path in mfcc_paths_set:
                    new_ids.append(curr_id)
                    new_labels.append(label)
            self.ids = new_ids
            self.labels = new_labels
            assert len(self.ids) == len(self.labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        curr_id = self.ids[index]
        curr_path = os.path.join(self.mfcc_dir, curr_id+'.mfcc')
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

def make_loader(ids, labels, args, shuffle=True, batch_size=64):
    '''
    Args:
        features: list of file id strings (files contain x values)
        labels: list of 1-dim int np arrays
    '''
    # Build the DataLoaders
    kwargs = {'pin_memory': True, 'num_workers': args.num_workers} if torch.cuda.is_available() else {}
    dataset = ASRDataset(ids, labels)
    loader = DataLoader(dataset, collate_fn=speech_collate_fn, shuffle=shuffle, batch_size=batch_size, **kwargs)
    return loader