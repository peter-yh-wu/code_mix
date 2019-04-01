import torch
import random
from torch.utils import data
from vocab import Vocab, Vectors
from configs import DEVICE


class DataSet(data.Dataset):
    def __init__(self, vocab=None, examples=None, padding=True, sort=False, sort_key=None):
        super(DataSet, self).__init__()
        self.examples = examples if examples is not None else []
        self.vocab = vocab
        self.padding = padding

        if sort and sort_key is not None:
            self.examples.sort(key=sort_key)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        assert item < len(self), 'index out of range'
        return self.examples[item]

    def get_subset(self, start, end):
        assert start < end, 'start index should be less than end index'
        return self.examples[start:end]

    def add(self, example):
        self.examples.append(example)

    def collate(self, batch):
        texts, labels, idxs = zip(*batch)

        # [batch_size]
        lens = torch.LongTensor(
            [len(text) for text in texts]
        )
        max_len = max(lens)
        # [batch_size, max_len]
        texts = torch.LongTensor(
            [
                torch.cat((text, torch.full((max_len - len(text)), self.vocab.stoi['<pad>'])))
                if self.padding and len(text) < max_len else text
                for text in texts
                ]
        )
        labels = torch.LongTensor(labels)

        return texts.to(DEVICE), labels.to(DEVICE)


class BilingualDataSet(DataSet):
    def __init__(self, vocab, examples=None, padding=True, sort=False, sort_key=None):
        super(BilingualDataSet, self).__init__(vocab, examples, padding, sort, sort_key)

    def collate(self, batch):
        # [batch_size]
        lens = [len(text) for text in batch]
        max_len = max(lens)
        # [batch_size, max_len]
        texts = [text + (max_len - len(text)) * ['<pad>']
                 if self.padding and len(text) < max_len else text for text in batch]

        return texts

