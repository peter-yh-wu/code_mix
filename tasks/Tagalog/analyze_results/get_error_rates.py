'''
Code to e.g. calculate CER, WER, and top-k CER

Peter Wu
peterw1@andrew.cmu.edu
'''
import argparse
import csv
import numpy as np
import os

from autocorrect import spell
from nltk.metrics import edit_distance

from model_utils import *

def cer_from_transcripts(transcripts, ys, log_path=None, truncate=True, spaces='best'):
    '''
    Args:
        transcripts: list of strings

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
        if log_path is not None:
            with open(log_path, 'a') as ouf:
                ouf.write('dist: %.2f, norm_dist: %.2f\n' % (best_dist, best_norm))
        norm_dists.append(best_norm)
        dists.append(best_dist)
    return norm_dists, dists

def get_cer(transcripts_file, save_dir='output/baseline/v1'):
    '''
    Args:
        transcripts_file: .csv file containing transcripts
            (e.g. submission.csv)
        save_dir: directory containing transcripts_file, as well as where to 
            save CER results
    '''
    t0 = time.time()
    _, test_ys = load_fid_and_y_data('test') # 1-dim np array of strings
    CSV_PATH = os.path.join(save_dir, transcripts_file)
    transcripts = []
    with open(CSV_PATH, 'r') as csvfile:
        raw_csv = csv.reader(csvfile)
        for row in raw_csv:
            transcripts.append(row[1])
    t1 = time.time()
    print('loaded data (%.2f seconds)' % (t1-t0))

    cer_log_path = os.path.join(save_dir, 'cer_log.txt')
    norm_dists, dists = cer_from_transcripts(transcripts, ys, log_path=cer_log_path)

    cers_path = os.path.join(save_dir, 'cers.npy')
    np.save(cers_path, norm_dists)
    
    print('avg CER:', np.mean(norm_dists))

def get_topk_cer(beam_file, save_dir='output/baseline/beam'):
    '''
    Args:
        beam_file: .csv file containing beam results
        save_dir: directory containing beam_file, as well as where to save
            CER results
    '''
    t0 = time.time()
    test_ys = load_y_data('test') # 1-dim np array of strings
    CSV_PATH = os.path.join(save_dir, beam_file)
    
    raw_ids = []
    raw_beams = [] # list of list of strings
    
    with open(CSV_PATH, 'r') as csvfile:
        raw_csv = csv.reader(csvfile)
        for row in raw_csv:
            raw_ids.append(row[0])
            raw_beams.append(row[1])
    t1 = time.time()
    print('loaded data (%.2f seconds)' % (t1-t0))

    num_beams = 0
    for curr_id in raw_ids:
        if curr_id == raw_ids[0]:
            num_beams += 1
        else:
            break

    test_ys_rep = []
    for y in test_ys:
        test_ys_rep += [y]*num_beams

    print('computing CER for all test samples (at %.2f seconds)' % (t1-t0))

    CER_LOG_PATH = os.path.join(save_dir, 'cer_log.txt')
    raw_norm_dists, raw_dists = cer_from_transcripts(raw_beams, test_ys_rep, log_path=CER_LOG_PATH)
    CER_PATH = os.path.join(save_dir, 'test_cer.npy')
    DIST_PATH = os.path.join(save_dir, 'test_dist.npy')
    raw_norm_dists = np.array(raw_norm_dists)
    raw_dists = np.array(raw_dists)
    norm_dists = raw_norm_dists.reshape((len(test_ys), num_beams))
    dists = raw_dists.reshape((len(test_ys), num_beams))
    np.save(CER_PATH, norm_dists)
    np.save(DIST_PATH, dists)
    
    min_cers = np.min(norm_dists, 1) # shape: (num_ys,)
    min_idxs = np.argmin(norm_dists, 1) # shape: (num_ys,)
    MIN_CERS_PATH = os.path.join(save_dir, 'min_cers.npy')
    MIN_IDXS_PATH = os.path.join(save_dir, 'min_idxs.npy')
    np.save(MIN_CERS_PATH, min_cers)
    np.save(MIN_IDXS_PATH, min_idxs)
    
    print('avg top-k CER:', np.mean(min_cers))

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
        fid = l[0]
        y = ' '.join(l[1:])
        ids.append(fid)
        ys.append(y)
    return ids, np.array(ys)

def closest_word(word, vocab, threshold=5, sub_thres=2):
    '''Finds closest word in the vocabulary (w.r.t. edit distance)

    Returns 2 words if no closest word found
    '''
    best_word = word
    best_dist = float("inf")
    prefix_len_best = float("inf")
    for vocab_word in vocab:
        curr_dist = edit_distance(word, vocab_word)
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_word = vocab_word
            prefix_len_best = len(os.path.commonprefix([word, vocab_word]))
        elif curr_dist == best_dist and abs(len(best_word)-len(word)) > abs(len(vocab_word)-len(word)):
            prefix_len_vocab = len(os.path.commonprefix([word, vocab_word]))
            if prefix_len_best < prefix_len_vocab:
                best_word = vocab_word
                prefix_len_best = prefix_len_vocab
    if best_dist > 5: # margin of error is sub_thres for each subword
        for i in range(len(word)-1):
            word1 = word[:i+1]
            word2 = word[i+1:]
            curr_dist = float("inf")
            vocab_word1 = word1
            for vocab_word in vocab:
                if word1 == vocab_word:
                    vocab_word1 = vocab_word
                    curr_dist = 0
                    break
                dist1 = edit_distance(word1, vocab_word)
                if dist1 < curr_dist:
                    vocab_word1 = vocab_word
                    curr_dist = dist1
            vocab_word2 = word2
            if curr_dist <= sub_thres:
                curr_dist2 = float("inf")
                for vocab_word in vocab:
                    if word2 == vocab_word:
                        vocab_word2 = vocab_word
                        curr_dist2 = 0
                        break
                    dist2 = edit_distance(word2, vocab_word)
                    if dist2 < curr_dist2:
                        vocab_word2 = vocab_word
                        curr_dist2 = dist2
                curr_dist += curr_dist2
                if curr_dist < best_dist:
                    best_word = vocab_word1+' '+vocab_word2
                    best_dist = curr_dist
    return best_word

def mk_map(vocab):
    '''
    Words are mapped to unique unicode characters

    Args:
        vocab: set of words (each word is a string)
    
    Return:
        wmap: dict, i.e. {str: new_str}, where each str is a token
    '''
    curr_unicode_num = 0
    wmap = {}
    for w in vocab:
        curr_unicode_ch = chr(curr_unicode_num)
        while curr_unicode_ch in vocab:
            curr_unicode_num += 1
            curr_unicode_ch = chr(curr_unicode_num)
        wmap[w] = curr_unicode_ch
        curr_unicode_num += 1
    return wmap

def map_lines(lines, wmap):
    new_lines = []
    for l in lines:
        l_list = l.split()
        new_l_list = []
        for w in l_list:
            new_l_list.append(wmap[w])
        new_l = ' '.join(new_l_list)
        new_lines.append(new_l)
    return new_lines

def get_wer(transcripts_file, save_dir='output/baseline/v1'):
    '''
    Args:
        transcripts_file: .csv file containing transcripts
            (e.g. submission.csv)
        save_dir: directory containing transcripts_file, as well as where to 
            save CER results
    '''
    t0 = time.time()
    _, test_ys = load_fid_and_y_data('test') # 1-dim np array of strings
    CSV_PATH = os.path.join(save_dir, transcripts_file)
    transcripts = []
    with open(CSV_PATH, 'r') as csvfile:
        raw_csv = csv.reader(csvfile)
        for row in raw_csv:
            transcripts.append(row[1])
    t1 = time.time()
    print('loaded data (%.2f seconds)' % (t1-t0))

    AUTOCORRECT_PATH = os.path.join(save_dir, 'transcript_autoc.txt')
    PROX_PATH = os.path.join(save_dir, 'transcript_prox.txt')
    AUTOC_PROX_PATH = os.path.join(save_dir, 'transcript_autoc_prox.txt')

    test_vocab = set()
    for test_y in test_ys:
        test_y_list = test_y.split()
        for word in test_y_list:
            test_vocab.add(word)

    if not os.path.exists(AUTOCORRECT_PATH) or not os.path.exists(PROX_PATH) or not os.path.exists(AUTOC_PROX_PATH):
        t1 = time.time()
        print('generating transcripts (at %.2f seconds)' % (t1-t0))

        transcript_lists = [l.split() for l in transcripts_spaced]

        new_transcripts_autoc = []
        new_transcripts_prox = []
        new_transcripts_autoc_prox = []
        for i, l_list in enumerate(transcript_lists):
            l_list_a = l_list.copy()
            l_list_p = l_list.copy()
            l_list_ap = l_list.copy()
            for word_i, w in enumerate(l_list):
                new_a_word = w
                new_p_word = w
                new_ap_word = w
                if w not in test_vocab:
                    new_a_word = spell(w)
                    new_p_word = closest_word(w, test_vocab)
                if new_a_word in test_vocab:
                    new_ap_word = new_a_word
                else:
                    new_ap_word = new_p_word
                l_list_a[word_i] = new_a_word
                l_list_p[word_i] = new_p_word
                l_list_ap[word_i] = new_ap_word
            new_l_a = ' '.join(l_list_a)
            new_l_p = ' '.join(l_list_p)
            new_l_ap = ' '.join(l_list_ap)
            new_transcripts_autoc.append(new_l_a)
            new_transcripts_prox.append(new_l_p)
            new_transcripts_autoc_prox.append(new_l_ap)
            if (i+1) % 500 == 0:
                t1 = time.time()
                print('Processed %d Lines (at %.2f seconds)' % (i+1, t1-t0))

        with open(AUTOCORRECT_PATH, 'w+') as ouf:
            for curr_s in new_transcripts_autoc:
                ouf.write('%s\n' % curr_s)

        with open(PROX_PATH, 'w+') as ouf:
            for curr_s in new_transcripts_prox:
                ouf.write('%s\n' % curr_s)

        with open(AUTOC_PROX_PATH, 'w+') as ouf:
            for curr_s in new_transcripts_autoc_prox:
                ouf.write('%s\n' % curr_s)
    else:
        with open(AUTOCORRECT_PATH, 'r') as inf:
            new_transcripts_autoc = inf.readlines()
        new_transcripts_autoc = [l.strip() for l in new_transcripts_autoc]
        with open(PROX_PATH, 'r') as inf:
            new_transcripts_prox = inf.readlines()
        new_transcripts_prox = [l.strip() for l in new_transcripts_prox]
        with open(AUTOC_PROX_PATH, 'r') as inf:
            new_transcripts_autoc_prox = inf.readlines()
        new_transcripts_autoc_prox = [l.strip() for l in new_transcripts_autoc_prox]    

    # ------------------------------------------
    # mer when vocab is test vocab
    # i.e. for new_transcripts_prox and new_transcripts_autoc_prox
    t1 = time.time()
    print('calculating mer values (at %.2f seconds)' % (t1-t0))
    
    test_vocab.add(' ')
    test_map = mk_map(test_vocab)

    prox_vocab = test_vocab.copy()
    for l in new_transcripts_prox:
        l_list = l.split()
        for word in l_list:
            prox_vocab.add(word)
    prox_map = mk_map(prox_vocab)

    autoc_prox_vocab = test_vocab.copy()
    for l in new_transcripts_autoc_prox:
        l_list = l.split()
        for word in l_list:
            autoc_prox_vocab.add(word)
    autoc_prox_map = mk_map(autoc_prox_vocab)

    transcripts_prox_uni = map_lines(new_transcripts_prox, prox_map)
    test_ys_spaced_uni = map_lines(test_ys_spaced, prox_map)
    PROX_MER_LOG_PATH = os.path.join(save_dir, 'prox_mer_log.txt')
    PROX_MER_PATH = os.path.join(save_dir, 'prox_mer.npy')
    PROX_DIST_PATH = os.path.join(save_dir, 'prox_dist.npy')
    prox_norm_dists, prox_dists = cer_from_transcripts(transcripts_prox_uni, test_ys_spaced_uni, PROX_MER_LOG_PATH)
    np.save(PROX_MER_PATH, prox_norm_dists)
    np.save(PROX_DIST_PATH, prox_dists)
    print('prox avg mer:', np.mean(prox_norm_dists))
    t1 = time.time()
    print('At %.2f seconds' % (t1-t0))

    transcripts_autoc_prox_uni = map_lines(new_transcripts_autoc_prox, autoc_prox_map)
    test_ys_spaced_autoc_prox_uni = map_lines(test_ys_spaced, autoc_prox_map)
    AUTOC_PROX_MER_LOG_PATH = os.path.join(save_dir, 'autoc_prox_mer_log.txt')
    AUTOC_PROX_MER_PATH = os.path.join(save_dir, 'autoc_prox_mer.npy')
    AUTOC_PROX_DIST_PATH = os.path.join(save_dir, 'autoc_prox_dist.npy')
    autoc_prox_norm_dists, autoc_prox_dists = cer_from_transcripts(transcripts_autoc_prox_uni, test_ys_spaced_autoc_prox_uni, AUTOC_PROX_MER_LOG_PATH)
    np.save(AUTOC_PROX_MER_PATH, autoc_prox_norm_dists)
    np.save(AUTOC_PROX_DIST_PATH, autoc_prox_dists)
    print('autoc prox avg mer:', np.mean(autoc_prox_norm_dists))
    t1 = time.time()
    print('At %.2f seconds' % (t1-t0))

    # ------------------------------------------
    # mer when vocab includes autoc
    # i.e. for new_transcripts_autoc
    autoc_vocab = test_vocab.copy()
    for l in new_transcripts_autoc:
        l_list = l.split()
        for word in l_list:
            autoc_vocab.add(word)
    autoc_map = mk_map(autoc_vocab)

    transcripts_autoc_uni = map_lines(new_transcripts_autoc, autoc_map)
    test_ys_spaced_autoc_uni = map_lines(test_ys_spaced, autoc_map)
    AUTOC_MER_LOG_PATH = os.path.join(save_dir, 'autoc_mer_log.txt')
    AUTOC_MER_PATH = os.path.join(save_dir, 'autoc_mer.npy')
    AUTOC_DIST_PATH = os.path.join(save_dir, 'autoc_dist.npy')
    autoc_norm_dists, autoc_dists = cer_from_transcripts(transcripts_autoc_uni, test_ys_spaced_autoc_uni, AUTOC_MER_LOG_PATH)
    np.save(AUTOC_MER_PATH, autoc_norm_dists)
    np.save(AUTOC_DIST_PATH, autoc_dists)
    print('autoc avg mer:', np.mean(autoc_prox_norm_dists))
    t1 = time.time()
    print('At %.2f seconds' % (t1-t0))

def parse_args():
    parser = argparse.ArgumentParser()
    praser.add_argument('--file', type=str, default='submission.csv', help='csv file with transcripts')
    parser.add_argument('--save-directory', type=str, default='output/baseline/v1', help='output directory')
    parser.add_argument('--mode', type=str, default='wer', help='wer, cer, or topk')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.mode == 'wer':
        get_wer(args.file, save_dir=args.save_directory)
    elif args.mod == 'cer':
        get_cer(args.file, save_dir=args.save_directory)
    else:
        get_topk_cer(args.file, save_dir=args.save_directory)

if __name__ == '__main__':
    main()