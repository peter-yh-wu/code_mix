#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

import os
import re
import torch

import multiprocessing as mp
from glob import glob


def preprocess(words):
    # Lemmatize the comments for better match
    words = words.replace('~', '').replace('`', '').replace('!', '').replace('@', '').replace('#', ''). \
        replace('$', '').replace('%', '').replace('^', '').replace('&', '').replace('*', ''). \
        replace('(', '').replace(')', '').replace('-', '').replace('_', '').replace('+', ''). \
        replace('=', '').replace('{', '').replace('}', '').replace('[', '').replace(']', ''). \
        replace('|', '').replace('\'', '').replace(':', '').replace(';', '').replace('"', ''). \
        replace('\"', '').replace('<', '').replace('>', '').replace(',', '').replace('.', ''). \
        replace('?', '').replace('/', '')
    words = [word.lower() for word in words.split()]
    return words


def las_to_lm(sentence):
    text = []
    for token in sentence:
        if not is_chinese_word(token) \
                or (is_chinese_word(token) and len(token) == 1):
            text.append(token)
        else:
            tmp = ""
            for char in token:
                if is_chinese_word(char):
                    if len(tmp) > 0:
                        text.append(tmp)
                        tmp = ""
                    text.append(char)
                else:
                    tmp += char
    return ['<s>'] + text + ['<s>']


def read_seame_data(files):
    data = []
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                text = []
                for token in line.split()[3:]:
                    if not is_chinese_word(token) \
                            or (is_chinese_word(token) and len(token) == 1):
                        text.append(token)
                    else:
                        tmp = ""
                        for char in token:
                            if is_chinese_word(char):
                                if len(tmp) > 0:
                                    text.append(tmp)
                                    tmp = ""
                                text.append(char)
                            else:
                                tmp += char
                assert (all(len(word) == 1 for word in text if is_chinese_word(word)))
                if len(text) > 0 and not all([word in ['ZH', 'CS', 'EN'] for word in text]):
                    data.append([word for word in ['<s>'] + text + ["<s>"] if word not in ['ZH', 'CS', 'EN']])

    return data


def read_qg_data(files):
    data = []
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                text = []
                for token in line.split()[3:]:
                    if token in ['，', '。', '！', '？', '…', '~', '=']:
                        continue
                    if not is_chinese_word(token) \
                            or (is_chinese_word(token) and len(token) == 1):
                        text.append(token)
                    else:
                        tmp = ""
                        for char in token:
                            if is_chinese_word(char):
                                if len(tmp) > 0:
                                    text.append(tmp)
                                    tmp = ""
                                text.append(char)
                            else:
                                tmp += char
                assert (all(len(word) == 1 for word in text if is_chinese_word(word)))
                if len(text) > 0 and not all([word in ['ZH', 'CS', 'EN'] for word in text]):
                    data.append([word for word in ['<s>'] + text + ["<s>"] if word not in ['ZH', 'CS', 'EN']])

    return data


def read_opensub_data(data_path):
    with open(os.path.join(data_path, 'english.txt')) as f:
        lines = f.readlines()
        eng_data = [preprocess(line) for line in lines[:35000]]
    # with open(os.path.join(data_path, 'spanish.txt')) as f:
    #     lines = f.readlines()
    #     spa_data = [line.split()[1:] for line in lines[:35000]]
    train = eng_data[:30000]
    dev = eng_data[30000:]
    train_ids = [[1 for _ in range(len(sent))] for sent in train[:30000]]
    dev_ids = [[1 for _ in range(len(sent))] for sent in train[:5000]]
    return train, dev, train_ids, dev_ids


def read_miami_data(data_path):
    with open(os.path.join(data_path, 'train.txt'), 'r') as f:
        lines = f.readlines()
        train = [line.split()[1:] for line in lines]
    with open(os.path.join(data_path, 'train_lids.txt'), 'r') as f:
        lines = f.readlines()
        train_ids = [line.split()[1:] for line in lines]
    with open(os.path.join(data_path, 'test.txt'), 'r') as f:
        lines = f.readlines()
        test = [line.split()[1:] for line in lines]
    with open(os.path.join(data_path, 'test_lids.txt'), 'r') as f:
        lines = f.readlines()
        test_ids = [line.split()[1:] for line in lines]
    with open(os.path.join(data_path, 'dev.txt'), 'r') as f:
        lines = f.readlines()
        dev = [line.split()[1:] for line in lines]
    with open(os.path.join(data_path, 'dev_lids.txt'), 'r') as f:
        lines = f.readlines()
        dev_ids = [line.split()[1:] for line in lines]

    train.extend(test)
    train_ids.extend(test_ids)

    miami_dict = {'eng': [], 'spa': []}
    for uttr, ids in zip(train, train_ids):
        if len(uttr) != len(ids):
            continue
        else:
            for tok, _id in zip(uttr, ids):
                if _id == 'eng':
                    miami_dict['eng'].append(tok)
                elif _id == 'spa':
                    miami_dict['spa'].append(tok)
                elif _id == 'engspa':
                    miami_dict['eng'].append(tok)
                    miami_dict['spa'].append(tok)
                else:
                    continue

    return train, dev, test, train_ids, dev_ids, test_ids, miami_dict


def read_dataset(data_path, num_workers=1):
    data = []
    all_file_paths = glob(os.path.join(data_path, '**/*.txt'), recursive=True)
    num_files = len(all_file_paths)
    files_per_worker = num_files // num_workers

    pool = mp.Pool(processes=num_workers)

    extraction_result = pool.map(read_seame_data,
                                 (all_file_paths[start_idx:start_idx+files_per_worker]
                                  for start_idx in range(0, num_files, files_per_worker)))

    for result in extraction_result:
        data.extend(result)
    return data


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


def has_chinese_char(word, _from='\u4e00', _to='\u9fff'):
    return len(re.findall(r'[{}-{}]+'.format(_from, _to), word)) > 0


def is_chinese_word(char):
    ranges = [
        {"from": u"\u3300", "to": u"\u33ff"},  # compatibility ideographs
        {"from": u"\ufe30", "to": u"\ufe4f"},  # compatibility ideographs
        {"from": u"\uf900", "to": u"\ufaff"},  # compatibility ideographs
        {"from": u"\U0002F800", "to": u"\U0002fa1f"},  # compatibility ideographs
        {'from': u'\u3040', 'to': u'\u309f'},  # Japanese Hiragana
        {"from": u"\u30a0", "to": u"\u30ff"},  # Japanese Katakana
        {"from": u"\u2e80", "to": u"\u2eff"},  # cjk radicals supplement
        {"from": u"\u4e00", "to": u"\u9fff"},
        {"from": u"\u3400", "to": u"\u4dbf"},
    ]
    return any([has_chinese_char(char, range['from'], range['to']) for range in ranges])