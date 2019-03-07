import argparse
import torch

parser = argparse.ArgumentParser(description='Language model parameters.')
parser.add_argument('--epoch', help='training epochs', default=20)
parser.add_argument('--model', help='choose language model', default='lstm')

args = parser.parse_args()

# configurations
N = 2  # The length of the n-gram
EMB_SIZE = 128  # The size of the embedding
HID_SIZE = 128  # The size of the hidden layer

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 100
