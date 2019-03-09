'''
Script to run LAS model

To-do:
 - Levenshtein

Modified from LAS implementation by Sai Krishna Rallabandi (srallaba@andrew.cmu.edu)

Peter Wu
peterw1@andrew.cmu.edu
'''

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model_utils import *


class SequenceShuffle(nn.Module):
    # Performs pooling for pBLSTM
    def forward(self, seq):
        assert isinstance(seq, PackedSequence)
        padded, lens = pad_packed_sequence(seq)  # (L, BS, D)
        padded = padded.transpose(0, 1)
        if padded.size(1) % 2 > 0:
            padded = padded[:, :-1, :]
        padded = padded.contiguous()
        padded = padded.view(padded.size(0), padded.size(1) // 2, 2 * padded.size(2))
        padded = padded.transpose(0, 1)
        newlens = np.array(lens) // 2
        newseq = pack_padded_sequence(padded, newlens)
        return newseq


class AdvancedLSTM(nn.LSTM):
    # Class for learning initial hidden states when using LSTMs
    def __init__(self, *args, **kwargs):
        super(AdvancedLSTM, self).__init__(*args, **kwargs)
        bi = 2 if self.bidirectional else 1
        self.h0 = Variable(torch.zeros((bi, 1, self.hidden_size), dtype=torch.float32))
        self.c0 = Variable(torch.zeros((bi, 1, self.hidden_size), dtype=torch.float32))
        if torch.cuda.is_available():
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()

    def initial_state(self, n):
        return (
            self.h0.expand(-1, n, -1).contiguous(),
            self.c0.expand(-1, n, -1).contiguous()
        )

    def forward(self, x, hx=None):
        if hx is None:
            n = x.batch_sizes[0]
            hx = self.initial_state(n)
        return super(AdvancedLSTM, self).forward(x, hx=hx)


class pLSTM(AdvancedLSTM):
    # Pyramidal LSTM
    def __init__(self, *args, **kwargs):
        super(pLSTM, self).__init__(*args, **kwargs)
        self.shuffle = SequenceShuffle()

    def forward(self, x, hx=None):
        return super(pLSTM, self).forward(self.shuffle(x), hx=hx)

INPUT_DIM = 39

class EncoderModel(nn.Module):
    # Encodes utterances to produce keys and values
    def __init__(self, args):
        super(EncoderModel, self).__init__()
        self.rnns = nn.ModuleList()
        self.rnns.append(AdvancedLSTM(INPUT_DIM, args.encoder_dim, bidirectional=True))
        self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
        self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
        self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
        self.key_projection = nn.Linear(args.encoder_dim * 2, args.key_dim)
        self.value_projection = nn.Linear(args.encoder_dim * 2, args.value_dim)

    def forward(self, utterances, utterance_lengths):
        h = utterances

        # Sort and pack the inputs
        sorted_lengths, order = torch.sort(utterance_lengths, 0, descending=True)
        _, backorder = torch.sort(order, 0)
        h = h[:, order, :]
        h = pack_padded_sequence(h, sorted_lengths.data.cpu().numpy())

        # RNNs
        for rnn in self.rnns:
            h, _ = rnn(h)

        # Unpack and unsort the sequences
        h, output_lengths = pad_packed_sequence(h)
        h = h[:, backorder, :]
        output_lengths = torch.from_numpy(np.array(output_lengths))
        if backorder.data.is_cuda:
            output_lengths = output_lengths.cuda()
        output_lengths = output_lengths[backorder.data]

        # Apply key and value
        keys = self.key_projection(h)
        values = self.value_projection(h)

        return keys, values, output_lengths


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def gumbel_argmax(logits, dim):
    # Draw from a multinomial distribution efficiently
    return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]


class AdvancedLSTMCell(nn.LSTMCell):
    # Extend LSTMCell to learn initial state
    def __init__(self, *args, **kwargs):
        super(AdvancedLSTMCell, self).__init__(*args, **kwargs)
        self.h0 = Variable(torch.zeros((1, self.hidden_size), dtype=torch.float32))
        self.c0 = Variable(torch.zeros((1, self.hidden_size), dtype=torch.float32))
        if torch.cuda.is_available():
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()

    def initial_state(self, n):
        return (
            self.h0.expand(n, -1).contiguous(),
            self.c0.expand(n, -1).contiguous()
        )


def output_mask(maxlen, lengths):
    """
    Create a mask on-the-fly
    :param maxlen: length of mask
    :param lengths: length of each sequence
    :return: mask shaped (maxlen, len(lengths))
    """
    lens = lengths.unsqueeze(0)
    ran = torch.arange(0, maxlen, 1, out=lengths.new()).unsqueeze(1)
    mask = ran < lens
    return mask


def calculate_attention(keys, mask, queries):
    """
    Attention calculation
    :param keys: (N, L, key_dim)
    :param mask: (N, L)
    :param queries: (N, key_dim)
    :return: attention (N, L)
    """
    energy = torch.bmm(keys, queries.unsqueeze(2)).squeeze(2) * mask  # (N, L)
    energy = energy - (1 - mask) * 1e4  # subtract large number from padded region
    emax = torch.max(energy, 1)[0].unsqueeze(1)  # (N, L)
    eval = torch.exp(energy - emax) * mask  # (N, L)
    attn = eval / (eval.sum(1).unsqueeze(1))  # (N, L)
    return attn


def calculate_context(attn, values):
    """
    Context calculation
    :param attn:  (N, L)
    :param values: (N, L, value_dim)
    :return: Context (N, value_dim)
    """
    ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)  # (N, value_dim)
    return ctx


class DecoderModel(nn.Module):
    # Speller/Decoder
    def __init__(self, args, vocab_size):
        super(DecoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, args.decoder_dim)
        self.input_rnns = nn.ModuleList()
        self.input_rnns.append(AdvancedLSTMCell(args.decoder_dim + args.value_dim, args.decoder_dim))
        self.input_rnns.append(AdvancedLSTMCell(args.decoder_dim, args.decoder_dim))
        self.input_rnns.append(AdvancedLSTMCell(args.decoder_dim, args.decoder_dim))
        self.query_projection = nn.Linear(args.decoder_dim, args.key_dim)
        self.char_projection = nn.Sequential(
            nn.Linear(args.decoder_dim+args.value_dim, args.decoder_dim),
            nn.LeakyReLU(),
            nn.Linear(args.decoder_dim, vocab_size+1)
        )
        self.force_rate = args.teacher_force_rate
        self.char_projection[-1].weight = self.embedding.weight  # weight tying

    def forward_pass(self, input_t, keys, values, mask, ctx, input_states):
        # Embed the previous character
        embed = self.embedding(input_t)
        # Concatenate embedding and previous context
        ht = torch.cat((embed, ctx), dim=1)
        # Run first set of RNNs
        new_input_states = []
        for rnn, state in zip(self.input_rnns, input_states):
            ht, newstate = rnn(ht, state)
            new_input_states.append((ht, newstate))
        # Calculate query
        query = self.query_projection(ht)
        # Calculate attention
        attn = calculate_attention(keys=keys, mask=mask, queries=query)
        # Calculate context
        ctx = calculate_context(attn=attn, values=values)
        # Concatenate hidden state and context
        ht = torch.cat((ht, ctx), dim=1)
        # Run projection
        logit = self.char_projection(ht)
        # Sample from logits
        generated = gumbel_argmax(logit, 1)  # (N,)
        return logit, generated, ctx, attn, new_input_states

    def forward(self, inputs, input_lengths, keys, values, utterance_lengths, future=0):
        mask = Variable(output_mask(values.size(0), utterance_lengths).transpose(0, 1)).float()
        values = values.transpose(0, 1)
        keys = keys.transpose(0, 1)
        t = inputs.size(0)
        n = inputs.size(1)

        # Initial states
        input_states = [rnn.initial_state(n) for rnn in self.input_rnns]

        # Initial context
        h0 = input_states[-1][0]
        query = self.query_projection(h0)
        attn = calculate_attention(keys, mask, query)
        ctx = calculate_context(attn, values)

        # Decoder loop
        logits = []
        attns = []
        generateds = []
        for i in range(t):
            # Use forced or generated inputs
            if len(generateds) > 0 and self.force_rate < 1 and self.training:
                input_forced = inputs[i]
                input_gen = generateds[-1]
                input_mask = Variable(input_forced.data.new(*input_forced.size()).bernoulli_(self.force_rate))
                input_t = (input_mask * input_forced) + ((1 - input_mask) * input_gen)
            else:
                input_t = inputs[i]
            # Run a single timestep
            logit, generated, ctx, attn, input_states = self.forward_pass(
                input_t=input_t, keys=keys, values=values, mask=mask, ctx=ctx,
                input_states=input_states
            )
            # Save outputs
            logits.append(logit)
            attns.append(attn)
            generateds.append(generated)

        # For future predictions
        if future > 0:
            assert len(generateds) > 0
            input_t = generateds[-1]
            for _ in range(future):
                # Run a single timestep
                logit, generated, ctx, attn, input_states = self.forward_pass(
                    input_t=input_t, keys=keys, values=values, mask=mask, ctx=ctx,
                    input_states=input_states
                )
                # Save outputs
                logits.append(logit)
                attns.append(attn)
                generateds.append(generated)
                # Pass generated as next x
                input_t = generated

        # Combine all the outputs
        logits = torch.stack(logits, dim=0)  # (L, N, Vocab Size)
        attns = torch.stack(attns, dim=0)  # (L, N, T)
        generateds = torch.stack(generateds,dim=0)
        return logits, attns, generateds


class Seq2SeqModel(nn.Module):
    # Tie encoder and decoder together
    def __init__(self, args, vocab_size):
        super(Seq2SeqModel, self).__init__()
        self.encoder = EncoderModel(args)
        self.decoder = DecoderModel(args, vocab_size=vocab_size)
        self._state_hooks = {}

    def forward(self, utterances, utterance_lengths, chars, char_lengths, future=0):
        keys, values, lengths = self.encoder(utterances, utterance_lengths)
        logits, attns, generated = self.decoder(chars, char_lengths, keys, values, lengths, future=future)
        self._state_hooks['attention'] = attns.permute(1, 0, 2).unsqueeze(1)
        return logits, generated, char_lengths


def decode_output(output, charset):
    # Convert ints back to strings
    chars = []
    for o in output:
        if o == 0:
            break
        chars.append(charset[o - 1])
    return "".join(chars)


def generate_transcripts(args, model, loader, charset):
    # Create and yield transcripts
    for uarray, ulens, l1array, llens, l2array in loader:
        if args.cuda:
            uarray = uarray.cuda()
            ulens = ulens.cuda()
            l1array = l1array.cuda()
            llens = llens.cuda()
        uarray = Variable(uarray)
        ulens = Variable(ulens)
        l1array = Variable(l1array)
        llens = Variable(llens)

        logits, generated, lens = model(
            uarray, ulens, l1array, llens,
            future=args.generator_length)
        generated = generated.data.cpu().numpy()  # (L, BS)
        n = uarray.size(1)
        for i in range(n):
            transcript = decode_output(generated[:, i], charset)
            yield transcript


def write_transcripts(path, args, model, loader, charset):
    # Write CSV file
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        transcripts = generate_transcripts(args, model, loader, charset)
        for i, t in enumerate(transcripts):
            w.writerow([i+1, t])


class SequenceCrossEntropy(nn.CrossEntropyLoss):
    # Customized CrossEntropyLoss
    def __init__(self, *args, **kwargs):
        super(SequenceCrossEntropy, self).__init__(*args, reduce=False, **kwargs)

    def forward(self, prediction, target):
        logits, generated, sequence_lengths = prediction
        maxlen = logits.size(0)
        mask = Variable(output_mask(maxlen, sequence_lengths.data)).float()
        logits = logits * mask.unsqueeze(2)
        losses = super(SequenceCrossEntropy, self).forward(logits.view(-1, logits.size(2)), target.view(-1))
        loss = torch.sum(mask.view(-1) * losses) / logits.size(1)
        return loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--save-directory', type=str, default='output/baseline/v1', help='output directory')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--num-workers', type=int, default=2, metavar='N', help='number of workers')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--max-train', type=int, default=1000000000, help='max train')
    parser.add_argument('--max-dev', type=int, default=1000000000, help='max dev')
    parser.add_argument('--max-test', type=int, default=1000000000, help='max test')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='N', help='lr')
    parser.add_argument('--weight-decay', type=float, default=1e-5, metavar='N', help='weight decay')
    parser.add_argument('--teacher-force-rate', type=float, default=0.9, metavar='N', help='teacher forcing rate')

    parser.add_argument('--encoder-dim', type=int, default=256, metavar='N', help='hidden dimension')
    parser.add_argument('--decoder-dim', type=int, default=512, metavar='N', help='hidden dimension')
    parser.add_argument('--value-dim', type=int, default=128, metavar='N', help='hidden dimension')
    parser.add_argument('--key-dim', type=int, default=128, metavar='N', help='hidden dimension')
    parser.add_argument('--generator-length', type=int, default=250, metavar='N', help='maximum length to generate')

    return parser.parse_args()

def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print("Loading File IDs")
    train_ids, dev_ids, test_ids = load_ids()
    train_ids, dev_ids, test_ids = train_ids[:args.max_train], dev_ids[:args.max_dev], test_ids[:args.max_test]
    
    print("Loading X Data")
    train_xs, train_indices = load_x_data(train_ids)
    dev_xs, dev_indices = load_x_data(dev_ids)
    test_xs, test_indices = load_x_data(test_ids)

    print("Loading Y Data")
    train_ys = load_y_data(train_indices, 'train')
    dev_ys = load_y_data(dev_indices, 'dev')
    test_ys = load_y_data(test_indices, 'test')

    print("Building Charset")
    charset = build_charset(np.concatenate((train_ys, dev_ys), axis=0))
    charmap = make_charmap(charset)
    charcount = len(charset)

    print("Mapping Characters")
    trainchars = map_characters(train_ys, charmap) # list of 1-dim int np arrays
    devchars = map_characters(dev_ys, charmap) # list of 1-dim int np arrays
    
    print("Building Loader")
    dev_loader = make_loader(dev_xs, devchars, args, shuffle=True, batch_size=args.batch_size)
    train_loader = make_loader(train_xs, trainchars, args, shuffle=True, batch_size=args.batch_size)
    test_loader = make_loader(test_xs, None, args, shuffle=False, batch_size=args.batch_size)
    
    print("Building Model")
    model = Seq2SeqModel(args, vocab_size=charcount)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = SequenceCrossEntropy()
    
    print("Running")
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)
    CKPT_PATH = os.path.join(args.save_directory, 'model.ckpt')
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH))
    LOG_PATH = os.path.join(args.save_directory, 'log')
    with open(LOG_PATH, 'w+') as ouf:
        pass
    if args.cuda:
        model = model.cuda()

    best_val_loss = sys.maxsize
    prev_best_epoch = 0
    for e in range(args.epochs):
        print_log('Starting Epoch %d' % (e+1), LOG_PATH)

        # train
        model.train()
        optimizer.zero_grad()
        l = 0
        for i, t in enumerate(train_loader):
            uarray, ulens, l1array, llens, l2array = t
            if torch.min(ulens).item() > 8 and torch.min(llens).item() > 0:
                uarray, ulens, l1array, llens, l2array = Variable(uarray), \
                    Variable(ulens), Variable(l1array), Variable(llens), Variable(l2array)
                if torch.cuda.is_available():
                    uarray, ulens, l1array, llens, l2array = uarray.cuda(), \
                        ulens.cuda(), l1array.cuda(), llens.cuda(), l2array.cuda()
                prediction = model(uarray, ulens, l1array, llens)
                logits, generated, char_lengths = prediction
                loss = criterion(prediction, l2array)
                l += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
        print_log('Train Loss: %f' % (l/len(train_loader.dataset)), LOG_PATH)
        
        # val
        model.eval()
        with torch.no_grad():
            l = 0
            for i, t in enumerate(dev_loader):
                uarray, ulens, l1array, llens, l2array = t
                if torch.min(ulens).item() > 8 and torch.min(llens).item() > 0:
                    uarray, ulens, l1array, llens, l2array = Variable(uarray), \
                        Variable(ulens), Variable(l1array), Variable(llens), Variable(l2array)
                    if torch.cuda.is_available():
                        uarray, ulens, l1array, llens, l2array = uarray.cuda(), \
                            ulens.cuda(), l1array.cuda(), llens.cuda(), l2array.cuda()
                    prediction = model(uarray, ulens, l1array, llens)
                    loss = criterion(prediction, l2array)
                    l += loss.item()
            val_loss = l/len(dev_loader.dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                prev_best_epoch = e
            elif e - prev_best_epoch > args.patience:
                break
            print_log('Val Loss: %f' % val_loss, LOG_PATH)
        
        # log
        if (e+1) % 4 == 0:
            torch.save(model.state_dict(), CKPT_PATH)
            write_transcripts(
            path=os.path.join(args.save_directory, 'submission_%d.csv' % (e+1)),
            args=args, model=model, loader=test_loader, charset=charset)

    write_transcripts(
    path=os.path.join(args.save_directory, 'submission.csv'),
    args=args, model=model, loader=test_loader, charset=charset)

if __name__ == '__main__':
    main()

