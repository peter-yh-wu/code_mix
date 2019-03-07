def read_dataset(filename, vocab):
    with open(filename, "r") as f:
        for line in f:
            yield [vocab[x] for x in line.strip().split(" ")[1:] if x != '<v-noise>']

