'''
Code to e.g. calculate MER

Language ID code from Zimeng Qiu

Peter Wu
peterw1@andrew.cmu.edu
'''

import argparse
import csv
import itertools
import numpy as np
import os
import re
import sys
import time
import torch

from autocorrect import spell
from nltk.metrics import edit_distance
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from baseline import parse_args, Seq2SeqModel, write_transcripts
from model_utils import *


def is_english_word(word):
    """
    Decide if a token in a document is an valid English word.
    For this project, we define valid English words to be ASCII strings that
    contain only letters (both upper and lower case), single quotes ('),
    double quotes ("), and hyphens (-). Double quotes may only appear at the
    beginning or end of a token unless the beginning/end of a token is no-letter
    characters. Double quotes cannot appear in the middle of letter characters.
    (for example, "work"- is valid, but home"work and -bi"cycle- are not).
    Tokens cannot be empty.
    :param word: A token in document.
    :return: Boolean value, True or False
    """
    return all([char in ["\"", "\'", "-", "*", "~", "*", "."] or char.isalpha() for char in word])


def is_chinese_char(ch):
    curr_ord = ord(ch)
    if 11904 <= curr_ord and curr_ord <= 12031:
        return True
    elif 12352 <= curr_ord and curr_ord <= 12543:
        return True
    elif 13056 <= curr_ord and curr_ord <= 19903:
        return True
    elif 19968 <= curr_ord and curr_ord <= 40959:
        return True
    elif 63744 <= curr_ord and curr_ord <= 64255:
        return True
    elif 65072 <= curr_ord and curr_ord <= 65103:
        return True
    elif 194560 <= curr_ord and curr_ord <= 195103:
        return True
    else:
        return False


def closest_word(word, vocab, threshold=5):
    '''Finds closest word in the vocabulary (w.r.t. edit distance)

    Returns 2 words if no closest word found
    '''
    best_word = word
    best_dist = 1000000000
    prefix_len_best = 1000000000
    for vocab_word in vocab:
        curr_dist = edit_distance(word, vocab_word)
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_word = vocab_word
            prefix_len_best = os.path.commonprefix([word, vocab_word])
        elif curr_dist == best_dist:
            prefix_len_vocab = len(os.path.commonprefix([word, vocab_word]))
            if prefix_len_best < prefix_len_vocab:
                best_word = vocab_word
                prefix_len_best = prefix_len_vocab
    if best_dist > 5: # margin of error is 1 for each subword
        for i in range(len(word)-1):
            word1 = word[:i+1]
            word2 = word[i+1:]
            curr_dist = 0
            vocab_word1 = word1
            for vocab_word in vocab:
                if word1 == vocab_word:
                    vocab_word1 = vocab_word
                    break
                if edit_distance(word1, vocab_word) == 1:
                    vocab_word1 = vocab_word
                    curr_dist = 1
                    break
            vocab_word2 = word2
            if curr_dist <= 1:
                for vocab_word in vocab:
                    if word2 == vocab_word:
                        vocab_word2 = vocab_word
                        break
                    if edit_distance(word2, vocab_word) == 1:
                        vocab_word2 = vocab_word
                        curr_dist += 1
                        break
            if curr_dist < best_dist:
                best_word = word1+' '+word2
                best_dist = curr_dist
    return best_word


def mk_map(vocab):
    '''
    chinese characters are mapped to themselves and english characters are
    mapped to unique unicode characters

    Assumes that vocab is only comprised of single chinese characters and
    english words

    Args:
        vocab: set of strings
    
    Return:
        wmap: dict, i.e. {str: new_str}, where each str is a token
    '''
    curr_unicode_num = 0
    wmap = {}
    for w in vocab:
        if is_chinese_char(w):
            wmap[w] = w
        else:
            curr_unicode_ch = chr(curr_unicode_num)
            while curr_unicode_ch in vocab:
                curr_unicode_num += 1
                curr_unicode_ch = chr(curr_unicode_num)
            wmap[w] = curr_unicode_ch
            curr_unicode_num += 1
    return wmap

def map_lines(lines, wmap):
    '''
    Assumes that all words are either only composed of chinese characters or
    only composed of english letters
    '''
    new_lines = []
    for l in lines:
        l_list = l.split()
        new_l_list = []
        for w in l_list:
            if is_chinese_char(w[0]):
                new_l_list.append(w)
            else:
                new_l_list.append(wmap(w))
        new_l = ' '.join(new_l_list)
        new_lines.append(new_l)
    return new_lines


def main():
    t0 = time.time()
    test_ys = load_y_data('test') # 1-dim np array of strings
    SAVE_DIR = 'output/baseline/v1'
    CSV_PATH = os.path.join(SAVE_DIR, 'submission.csv')
    transcripts = []
    with open(CSV_PATH, 'r') as csvfile:
        raw_csv = csv.reader(csvfile)
        for row in raw_csv:
            transcripts.append(row[1])
    t1 = time.time()
    print('loaded data (%.2f seconds)' % (t1-t0))

    test_ys_spaced = []
    for test_y in test_ys:
        curr_s = ''
        next_ch = '' # for scope
        for s_i, ch in enumerate(test_y[:-1]):
            next_ch = test_y[s_i+1]
            curr_s += ch
            ch_is_chinese = is_chinese_char(ch)
            next_ch_is_chinese = is_chinese_char(next_ch)
            if (ch_is_chinese and not next_ch_is_chinese) or (next_ch_is_chinese and not ch_is_chinese):
                curr_s += ' '
        curr_s += next_ch
        test_ys_spaced.append(curr_s)

    test_ys_eng = []
    for test_y in test_ys:
        curr_s = ''
        for ch in test_y:
            if not is_chinese_char(ch):
                curr_s += ch
        test_ys_eng.append(curr_s)
    test_ys_eng = [l.strip() for l in test_ys_eng]

    transcripts_spaced = []
    for transcript in transcripts:
        curr_s = ''
        next_ch = '' # for scope
        for s_i, ch in enumerate(transcript[:-1]):
            next_ch = transcript[s_i+1]
            curr_s += ch
            ch_is_chinese = is_chinese_char(ch)
            next_ch_is_chinese = is_chinese_char(next_ch)
            if (ch_is_chinese and not next_ch_is_chinese) or (next_ch_is_chinese and not ch_is_chinese):
                curr_s += ' '
        curr_s += next_ch
        transcripts_spaced.append(curr_s)

    transcripts_eng = []
    for transcript in transcripts_spaced:
        curr_s = ''
        for ch in transcript:
            if not is_chinese_char(ch):
                curr_s += ch
        transcripts_eng.append(curr_s)

    # auto-correct run
    t1 = time.time()
    print('generating transcript_autoc (at %.2f seconds)' % (t1-t0))
    test_eng_vocab = set()
    for test_y in test_ys_eng:
        test_y_list = test_y.split()
        for word in test_y_list:
            test_eng_vocab.add(word)

    new_transcripts_autoc = []
    for i, transcript_eng in enumerate(transcripts_eng):
        transcript_eng_list = transcript_eng.split()
        curr_s = transcripts_spaced[i]
        curr_s_i = 0
        for eng_word in transcript_eng_list:
            curr_s_pre = curr_s[:curr_s_i]
            curr_s_suf = curr_s[curr_s_i:]
            new_word = eng_word
            if eng_word not in test_eng_vocab:
                new_word = spell(eng_word)
            curr_s = curr_s_pre+curr_s_suf.replace(eng_word, new_word, 1)
            curr_s_i = curr_s.find(eng_word, curr_s_i)+len(new_word)
        new_transcripts_autoc.append(curr_s)

    AUTOCORRECT_PATH = os.path.join(SAVE_DIR, 'transcript_autoc.txt')
    with open(AUTOCORRECT_PATH, 'w+') as ouf:
        for curr_s in new_transcripts_autoc:
            ouf.write('%s\n' % curr_s)

    # edit distance run
    t1 = time.time()
    print('generating transcript_prox (at %.2f seconds)' % (t1-t0))
    new_transcripts_prox = []
    for i, transcript_eng in enumerate(transcripts_eng):
        transcript_eng_list = transcript_eng.split()
        curr_s = transcripts_spaced[i]
        curr_s_i = 0
        for eng_word in transcript_eng_list:
            curr_s_pre = curr_s[:curr_s_i]
            curr_s_suf = curr_s[curr_s_i:]
            new_word = eng_word
            if eng_word not in test_eng_vocab:
                new_word = closest_word(eng_word, test_eng_vocab)
            curr_s = curr_s_pre+curr_s_suf.replace(eng_word, new_word, 1)
            curr_s_i = curr_s.find(eng_word, curr_s_i)+len(new_word)
        new_transcripts_prox.append(curr_s)

    PROX_PATH = os.path.join(SAVE_DIR, 'transcript_prox.txt')
    with open(PROX_PATH, 'w+') as ouf:
        for curr_s in new_transcripts_prox:
            ouf.write('%s\n' % curr_s)

    # edit distance plus auto-correct run
    t1 = time.time()
    print('generating transcript_autoc_prox (at %.2f seconds)' % (t1-t0))
    new_transcripts_autoc_prox = []
    for i, transcript_eng in enumerate(transcripts_eng):
        transcript_eng_list = transcript_eng.split()
        curr_s = transcripts_spaced[i]
        curr_s_i = 0
        for eng_word in transcript_eng_list:
            curr_s_pre = curr_s[:curr_s_i]
            curr_s_suf = curr_s[curr_s_i:]
            new_word = eng_word
            if eng_word not in test_eng_vocab:
                new_word = spell(eng_word)
                if new_word not in test_eng_vocab:
                    new_word = closest_word(eng_word, test_eng_vocab)
            curr_s = curr_s_pre+curr_s_suf.replace(eng_word, new_word, 1)
            curr_s_i = curr_s.find(eng_word, curr_s_i)+len(new_word)
        new_transcripts_autoc_prox.append(curr_s)

    AUTOC_PROX_PATH = os.path.join(SAVE_DIR, 'transcript_autoc_prox.txt')
    with open(AUTOC_PROX_PATH, 'w+') as ouf:
        for curr_s in new_transcripts_autoc_prox:
            ouf.write('%s\n' % curr_s)

    # ------------------------------------------
    # mer when vocab is test vocab
    # i.e. for new_transcripts_prox and new_transcripts_autoc_prox
    test_vocab = set() # composed of single chinese characters and english words
    num_test_eng = 0
    for test_y in test_ys_spaced:
        test_y_list = test_y.split()
        for word in test_y_list:
            if is_chinese_char(word[0]):
                for ch in word:
                    test_vocab.add(ch)
            else:
                test_vocab.add(word)
                num_test_eng += 1
    test_vocab.add(' ')
    test_map = mk_map(test_vocab)

    transcripts_prox_uni = map_lines(new_transcripts_prox, test_map)
    test_ys_spaced_uni = map_lines(test_ys_spaced, test_map)
    PROX_MER_LOG_PATH = os.path.join(SAVE_DIR, 'prox_mer_log.txt')
    PROX_MER_PATH = os.path.join(SAVE_DIR, 'prox_mer.npy')
    PROX_DIST_PATH = os.path.join(SAVE_DIR, 'prox_dist.npy')
    prox_norm_dists, prox_dists = cer_from_transcripts(transcripts_prox_uni, test_ys_spaced_uni, PROX_MER_LOG_PATH)
    np.save(PROX_MER_PATH, prox_norm_dists)
    np.save(PROX_DIST_PATH, prox_dists)
    print('prox avg cer:', np.mean(prox_norm_dists))

    transcripts_autoc_prox_uni = map_lines(new_transcripts_autoc_prox, test_map)
    AUTOC_PROX_MER_LOG_PATH = os.path.join(SAVE_DIR, 'autoc_prox_mer_log.txt')
    AUTOC_PROX_MER_PATH = os.path.join(SAVE_DIR, 'autoc_prox_mer.npy')
    AUTOC_PROX_DIST_PATH = os.path.join(SAVE_DIR, 'autoc_prox_dist.npy')
    autoc_prox_norm_dists, autoc_prox_dists = cer_from_transcripts(transcripts_autoc_prox_uni, test_ys_spaced_uni, AUTOC_PROX_MER_LOG_PATH)
    np.save(AUTOC_PROX_MER_PATH, autoc_prox_norm_dists)
    np.save(AUTOC_PROX_DIST_PATH, autoc_prox_dists)
    print('autoc prox avg cer:', np.mean(autoc_prox_norm_dists))

    # ------------------------------------------
    # mer when vocab includes autoc
    # i.e. for new_transcripts_autoc
    autoc_vocab = test_vocab.copy()
    for l in new_transcripts_autoc:
        l_list = l.split()
        for word in l_list:
            if is_chinese_char(word[0]):
                for ch in word:
                    autoc_vocab.add(ch)
            else:
                autoc_vocab.add(word)
    autoc_map = mk_map(autoc_vocab)

    transcripts_autoc_uni = map_lines(new_transcripts_autoc, autoc_map)
    test_ys_spaced_autoc_uni = map_lines(test_ys_spaced, autoc_map)
    AUTOC_MER_LOG_PATH = os.path.join(SAVE_DIR, 'autoc_mer_log.txt')
    AUTOC_MER_PATH = os.path.join(SAVE_DIR, 'autoc_mer.npy')
    AUTOC_DIST_PATH = os.path.join(SAVE_DIR, 'autoc_dist.npy')
    autoc_norm_dists, autoc_dists = cer_from_transcripts(transcripts_autoc_uni, test_ys_spaced_autoc_uni, AUTOC_MER_LOG_PATH)
    np.save(AUTOC_MER_PATH, autoc_norm_dists)
    np.save(AUTOC_DIST_PATH, autoc_dists)
    print('autoc avg cer:', np.mean(autoc_prox_norm_dists))

if __name__ == '__main__':
    main()