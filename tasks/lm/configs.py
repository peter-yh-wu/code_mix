import argparse
import torch
from datetime import datetime

parser = argparse.ArgumentParser(description='Language model parameters.')
parser.add_argument('--epoch', help='maximum training epochs', type=int, default=20)
parser.add_argument('--model', help='choose language model', default='lstm')
parser.add_argument('--batch', help='batch size', type=int, default=1)
parser.add_argument('--embed_en', help='pre-trained word embedding for English')
parser.add_argument('--embed_cn', help='pre-trained word embedding for Chinese')
parser.add_argument('--hidden', help='LSTM hidden unit size', type=int, default=128)
parser.add_argument('--embed', help='word embedding vector size', type=int, default=300)
parser.add_argument('--ngram', help='ngram language model', type=int, default=1)
parser.add_argument('--maxlen', help='maximum length of sentence', type=int, default=30)
parser.add_argument('--optim', help='optimizer, Adadelta, Adam or SGD', default='adadelta')
parser.add_argument('--dp', help='dropout rate, float number from 0 to 1.', default=0.5, type=float)
parser.add_argument('--mode', help='train/test', default='train')
parser.add_argument('--nworkers', help='number of workers for loading dataset', default=4)
parser.add_argument('--lr', help='initial learning rate', type=float, default=0.01)
parser.add_argument('--mm', help='momentum', type=float, default=0.9)
parser.add_argument('--clip', help='gradient clipping', type=float, default=0.25)

args = parser.parse_args()

# running configurations
log_dir = 'log/'
timestamp = datetime.now().strftime('%m%d-%H%M%S')
names = ('train', 'dev') if args.mode == 'train' else ('test',)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


