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
    dataset = read_dataset(args.data, num_workers=12)
    dataset = dataset[: int(len(dataset) * args.subset)]
    train = dataset[: int(len(dataset) * 0.8)]
    dev = dataset[int(len(dataset) * 0.8) + 1: -1]
    print(f'  Total samples:  {len(dataset)}')
    print(f'    Training:     {len(train)}')
    print(f'    Dev:          {len(dev)}')

    print('Building vocabulary')
    vocab = Vocab(train)

    print('=== Vocabulary (START) ===')
    for idx in range(len(vocab)):
        word = vocab.itos[idx]
        if len(word) < 1:
            print(f'[{idx}] (EMPTY)')
        else:
            print(f'[{idx}] ({"CN" if is_chinese_word(word) else "EN"})  {word}')
    print('=== Vocabulary (END) ===')

    print(f'  Vocabulary size:  {len(vocab)}')

if __name__ == '__main__':
    main()