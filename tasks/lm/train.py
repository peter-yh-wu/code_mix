import math
import time
import random
import logging
import torch
import torch.nn as nn

from collections import defaultdict
from torch.autograd import Variable
from lm import FNNLM, DualLSTM, LSTMLM
from utils import *
from configs import *


logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# convert a (nested) list of int into a pytorch Variable
def convert_to_variable(words):
    var = Variable(torch.LongTensor(words))
    if USE_CUDA:
        var = var.cuda()

    return var


# A function to calculate scores for one value
def calc_score_of_histories(words, model):
    # This will change from a list of histories, to a pytorch Variable whose data type is LongTensor
    words_var = convert_to_variable(words)
    logits = model(words_var)
    return logits


# Calculate the loss value for the entire sentence
def calc_sent_loss(sent, model):
    # The initial history is equal to end of sentence symbols
    hist = [S] * N
    # Step through the sentence, including the end of sentence token
    all_histories = []
    all_targets = []
    for next_word in sent + [S]:
        all_histories.append(list(hist))
        all_targets.append(next_word)
        hist = hist[1:] + [next_word]

    logits = calc_score_of_histories(all_histories, model)
    loss = nn.functional.cross_entropy(logits, convert_to_variable(all_targets), size_average=False)

    return loss


# Generate a sentence
def generate_sent(model):
    hist = [S] * N
    sent = []
    while True:
        logits = calc_score_of_histories([hist], model)
        prob = nn.functional.softmax(logits)
        next_word = prob.multinomial(1).data[0, 0]
        if next_word == S or len(sent) == MAX_LEN:
            break
        sent.append(next_word)
        hist = hist[1:] + [next_word]
    return sent


if __name__ == '__main__':
    w2i = defaultdict(lambda: len(w2i))
    # w2i_eng = defaultdict(lambda: len(w2i_eng))
    # w2i_chn = defaultdict(lambda: len(w2i_chn))

    S = w2i['<s>']
    UNK = w2i['<unk>']

    # for vocab in [w2i, w2i_eng, w2i_chn]:
    #     S = vocab["<s>"]
    #     UNK = vocab["<unk>"]

    # Read in the data
    train = list(read_dataset("SEAME-dev-set/dev_man/text", w2i))
    w2i = defaultdict(lambda: UNK, w2i)
    dev = list(read_dataset("SEAME-dev-set/dev_sge/text", w2i))
    i2w = {v: k for k, v in w2i.items()}
    n_words = len(w2i)

    # Initialize the model and the optimizer
    if args.model.lower() == 'lstm':
        model = LSTMLM(n_words=n_words, emb_size=EMB_SIZE,
                       hidden_dim=HID_SIZE, num_hist=N, dropout=0.2, n_layers=1)
    elif args.model.lower() == 'fnn':
        model = FNNLM(n_words=n_words, emb_size=EMB_SIZE,
                      hid_size=HID_SIZE, num_hist=N, dropout=0.2)
    else:
        model = LSTMLM(n_words=n_words, emb_size=EMB_SIZE,
                       hidden_dim=HID_SIZE, num_hist=N, dropout=0.2, n_layers=1)

    if USE_CUDA:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    last_dev = 1e20
    best_dev = 1e20

    for ITER in range(20):
        # Perform training
        random.shuffle(train)
        # set the model to training mode
        model.train()
        train_words, train_loss = 0, 0.0
        start = time.time()
        for sent_id, sent in enumerate(train):
            my_loss = calc_sent_loss(sent, model)
            train_loss += my_loss.data
            train_words += len(sent)
            optimizer.zero_grad()
            # my_loss.backward(retain_graph=True)
            my_loss.backward()
            optimizer.step()
            if (sent_id + 1) % 500 == 0:
                logger.info(
                    "--finished %r sentences (word/sec=%.2f)" % (sent_id + 1, train_words / (time.time() - start)))
        logger.info("iter %r: train loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (
            ITER, train_loss / train_words, math.exp(train_loss / train_words), train_words / (time.time() - start)))

        # Evaluate on dev set
        # set the model to evaluation mode
        model.eval()
        dev_words, dev_loss = 0, 0.0
        start = time.time()
        for sent_id, sent in enumerate(dev):
            my_loss = calc_sent_loss(sent, model)
            dev_loss += my_loss.data
            dev_words += len(sent)

        # Keep track of the development accuracy and reduce the learning rate if it got worse
        if last_dev < dev_loss and type(optimizer) is not torch.optim.Adam:
            optimizer.learning_rate /= 2
        last_dev = dev_loss

        # Keep track of the best development accuracy, and save the model only if it's the best one
        if best_dev > dev_loss:
            torch.save(model, "model.pt")
            best_dev = dev_loss

        # Save the model
        logger.info("iter %r: dev loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (
            ITER, dev_loss / dev_words, math.exp(dev_loss / dev_words), dev_words / (time.time() - start)))

        # Generate a few sentences
        for _ in range(5):
            sent = generate_sent(model)
            print(" ".join([i2w[x.item()] for x in sent]))