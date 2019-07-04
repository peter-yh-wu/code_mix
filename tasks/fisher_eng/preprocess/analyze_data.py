import itertools
import os

def build_charset(utterances):
    # Create a character set
    chars = set(itertools.chain.from_iterable(utterances))
    chars = list(chars)
    chars.sort()
    return chars

def get_charset(path):
    with open(path, 'r') as inf:
        lines = inf.readlines()
    all_words = []
    for l in lines:
        l = l.strip()
        l_list = l.split()
        word_list = l_list[1:]
        words = ' '.join(word_list)
        all_words.append(words)
    charset = build_charset(all_words)
    return set(charset)

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    split_dir = os.path.join(parent_dir, 'split')

    train_file = 'train.txt'
    dev_file = 'dev.txt'
    test_file = 'test.txt'

    train_path = os.path.join(split_dir, train_file)
    dev_path = os.path.join(split_dir, dev_file)
    test_path = os.path.join(split_dir, test_file)

    train_charset = get_charset(train_path)
    dev_charset = get_charset(dev_path)
    test_charset = get_charset(test_path)
    all_charset = train_charset.union(dev_charset).union(test_charset)
    print(len(all_charset))
    print(all_charset)

if __name__ == '__main__':
    main()