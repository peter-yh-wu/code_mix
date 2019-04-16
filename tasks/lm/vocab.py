#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Modified from torchtext
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

from __future__ import unicode_literals
from collections import defaultdict

import os
import logging
import torch
import six
import gzip

from tqdm import tqdm
from collections import Counter
from functools import reduce
from utils.data import is_english_word, is_chinese_word

logger = logging.getLogger(__name__)
torch.manual_seed(10707)


def _default_unk_index():
    return 0


def _default_s_index():
    return 1


def _rand_int(dim):
    return torch.randn(dim) * torch.rsqrt(dim)


def _uni_int(dim):
    return torch.rand(dim) * torch.rsqrt(dim)


def _zero_int(dim):
    return torch.zeros((1, dim))


def _infer_shape(f):
    num_lines, vector_dim = 0, None
    for line in f:
        if vector_dim is None:
            row = line.rstrip().split(b" ")
            vector = row[1:]
            # Assuming word, [vector] format
            if len(vector) > 2:
                # The header present in some (w2v) formats contains two elements.
                vector_dim = len(vector)
                num_lines += 1  # First element read
        else:
            num_lines += 1
    f.seek(0)
    return num_lines, vector_dim


class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """
    def __init__(self, words, max_size=None, min_freq=1, specials=['<s>', '<pad>'],
                 vectors=None, specials_first=True, filter_func=None, unk_init=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            words: Corpus for building vocabulary.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            specials_first: Whether to add special tokens into the vocabulary at first.
                If it is False, they are added into the vocabulary at last.
                Default: True.
        """
        self.max_size = max_size
        self.min_freq = min_freq
        self.specials = specials
        self.specials_first = specials_first
        self.pre_trained = vectors
        self.vectors = None
        self.freqs = None
        if filter_func == 'eng':
            self.filter_func = is_english_word
        elif filter_func == 'chn':
            self.filter_func = is_chinese_word
        else:
            self.filter_func = None

        self.itos = []
        if specials_first:
            self.itos = specials

        if '<unk>' in specials:  # hard-coded for now
            self.stoi = defaultdict(_default_unk_index)
        else:
            self.stoi = defaultdict()

        assert words is not None
        self.build(words)
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init

    def build(self, words):
        counter = Counter(reduce(lambda x, y: x + y, words))
        min_freq = max(self.min_freq, 1)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in self.specials:
            del counter[tok]

        self.max_size = None if self.max_size is None else self.max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == self.max_size:
                break
            if self.filter_func is None or self.filter_func(word):
                self.itos.append(word)

        if not self.specials_first:
            self.itos.extend(self.specials)

        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        if self.pre_trained is not None:
            self.load_vectors(self.pre_trained)

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.stoi:
                return self.stoi[item]
            else:
                self.itos.append(item)
                idx = len(self.itos) - 1
                self.stoi[item] = idx
                if self.vectors is not None:
                    self.vectors[idx] = self.unk_init(torch.Tensor(self.vectors.dim))
        else:
            return self.itos[item]

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

    def load_vectors(self, vectors):
        """
        Arguments:
            vectors: one of or a list containing instantiations of the
                GloVe, CharNGram, or Vectors classes. Alternatively, one
                of or a list of available pretrained vectors:
        """
        if not isinstance(vectors, list):
            vectors = [vectors]
        tot_dim = sum(v.dim for v in vectors)
        self.vectors = torch.Tensor(len(self), tot_dim)
        for i, token in enumerate(self.itos):
            start_dim = 0
            for v in vectors:
                end_dim = start_dim + v.dim
                self.vectors[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert (start_dim == tot_dim)


class Vectors(object):
    """
    From torchtext.vocab.Vectors
    """
    def __init__(self, name, cache=None,
                 url=None, unk_init=None, max_vectors=None):
        """
        Arguments:
           name: name of the file that contains the vectors
           cache: directory for cached vectors
           url: url for download if vectors not found in cache
           unk_init (callback): by default, initialize out-of-vocabulary word vectors
               to zero vectors; can be any function that takes in a Tensor and
               returns a Tensor of the same size
           max_vectors (int): this can be used to limit the number of
               pre-trained vectors loaded.
               Most pre-trained vector sets are sorted
               in the descending order of word frequency.
               Thus, in situations where the entire set doesn't fit in memory,
               or is not needed for another reason, passing `max_vectors`
               can limit the size of the loaded set.
        """
        cache = '.vector_cache' if cache is None else cache
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        self.cache(name, cache, url=url, max_vectors=max_vectors)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(self.dim))

    def cache(self, name, cache, url=None, max_vectors=None):
        if os.path.isfile(name):
            path = name
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix
        else:
            path = os.path.join(cache, name)
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = path + file_suffix

        if not os.path.isfile(path_pt):
            # if not os.path.isfile(path) and url:
            #     logger.info('Downloading vectors from {}'.format(url))
            #     if not os.path.exists(cache):
            #         os.makedirs(cache)
            #     dest = os.path.join(cache, os.path.basename(url))
            #     if not os.path.isfile(dest):
            #         with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
            #             try:
            #                 urlretrieve(url, dest, reporthook=reporthook(t))
            #             except KeyboardInterrupt as e:  # remove the partial zip file
            #                 os.remove(dest)
            #                 raise e
            #     logger.info('Extracting vectors into {}'.format(cache))
            #     ext = os.path.splitext(dest)[1][1:]
            #     if ext == 'zip':
            #         with zipfile.ZipFile(dest, "r") as zf:
            #             zf.extractall(cache)
            #     elif ext == 'gz':
            #         if dest.endswith('.tar.gz'):
            #             with tarfile.open(dest, 'r:gz') as tar:
            #                 tar.extractall(path=cache)
            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            logger.info("Loading vectors from {}".format(path))
            ext = os.path.splitext(path)[1][1:]
            if ext == 'gz':
                open_file = gzip.open
            else:
                open_file = open

            vectors_loaded = 0
            with open_file(path, 'rb') as f:
                num_lines, dim = _infer_shape(f)
                if not max_vectors or max_vectors > num_lines:
                    max_vectors = num_lines

                itos, vectors, dim = [], torch.zeros((max_vectors, dim)), None

                for line in tqdm(f, total=num_lines):
                    # Explicitly splitting on " " is important, so we don't
                    # get rid of Unicode non-breaking spaces in the vectors.
                    entries = line.rstrip().split(b" ")

                    word, entries = entries[0], entries[1:]
                    if dim is None and len(entries) > 1:
                        dim = len(entries)
                    elif len(entries) == 1:
                        logger.warning("Skipping token {} with 1-dimensional "
                                       "vector {}; likely a header".format(word, entries))
                        continue
                    elif dim != len(entries):
                        raise RuntimeError(
                            "Vector for token {} has {} dimensions, but previously "
                            "read vectors have {} dimensions. All vectors must have "
                            "the same number of dimensions.".format(word, len(entries),
                                                                    dim))

                    try:
                        if isinstance(word, six.binary_type):
                            word = word.decode('utf-8')
                    except UnicodeDecodeError:
                        logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                        continue

                    vectors[vectors_loaded] = torch.tensor([float(x) for x in entries])
                    vectors_loaded += 1
                    itos.append(word)

                    if vectors_loaded == max_vectors:
                        break

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)



