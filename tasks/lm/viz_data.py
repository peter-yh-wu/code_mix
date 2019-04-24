#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Salvador Medina <salvadom@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

from configs import *
from utils.data import *
from vocab import Vocab

def main():
    print('Loading dataset')
    dataset = read_dataset(args.data)
    dataset = dataset[: int(len(dataset) * args.subset)]
    train = dataset[: int(len(dataset) * 0.8)]
    dev = dataset[int(len(dataset) * 0.8) + 1: -1]
    vocab = Vocab(train)

    print('=== Vocabulary ===')
    for idx in range(len(vocab)):
        word = vocab.itos[idx]
        print(f'[{idx}] ({"CN" if has_chinese_char(word) else "EN"})  {word}')

    print(f'  Training samples: {len(train)}')
    print(f'  Dev samples:      {len(dev)}')
    print(f'  Vocabulary size:  {len(vocab)}')

if __name__ == '__main__':
    main()