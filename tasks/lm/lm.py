import torch
import torch.nn as nn
import torch.nn.functional as F


class FNNLM(nn.Module):
    """
    Feed-forward Neural Network Language Model
    """
    def __init__(self, n_words, emb_size, hid_size, num_hist, dropout):
        super(FNNLM, self).__init__()
        self.embedding = nn.Embedding(n_words, emb_size)
        self.fnn = nn.Sequential(
            nn.Linear(num_hist*emb_size, hid_size), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_size, n_words)
        )

    def forward(self, words):
        emb = self.embedding(words)       # 3D Tensor of size [batch_size x num_hist x emb_size]
        feat = emb.view(emb.size(0), -1)  # 2D Tensor of size [batch_size x (num_hist*emb_size)]
        logit = self.fnn(feat)            # 2D Tensor of size [batch_size x nwords]

        return logit


class LSTMLM(nn.Module):
    """
    Feed-forward Neural Network Language Model
    """
    def __init__(self, n_words, emb_size, hidden_dim, n_layers, num_hist, dropout, bidirection=2):
        super(LSTMLM, self).__init__()
        self.hidden_dim = hidden_dim
        self.ngram = num_hist
        self.bidirection = bidirection
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_words, emb_size)

        self.lstm = nn.LSTM(input_size=num_hist*emb_size,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True if self.bidirection == 2 else False,
                            dropout=dropout)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Sequential(
            nn.Linear(self.bidirection*hidden_dim, n_words),
            nn.ReLU(),
            nn.Linear(n_words, n_words)
        )
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.cat((torch.zeros(self.n_layers, 1, self.hidden_dim),
                          torch.zeros(self.n_layers, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        prediction = self.linear(lstm_out.squeeze(1).view(len(sentence), -1))
        return F.log_softmax(prediction, dim=1)


class DualLSTM(nn.Module):
    """
    Dual Bidirectional LSTM Language Model
    """
    def __init__(self, embeddings, freeze, input_dim, hidden_dim, n_layers,
                 output_dim, vocab_size, dropout, multilingual=2, bidirectional=True):
        super(DualLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=freeze)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.ModuleList(
            [
                nn.LSTM(input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=n_layers,
                        bidirectional=bidirectional,
                        dropout=dropout)
                for lang in range(multilingual)
            ])

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        lstm_out, self.hidden = [lstm(embeds.view(len(sentence), 1, -1), self.hidden)
                                 for lstm in self.lstm]
        prediction = self.linear(lstm_out.view(len(sentence), -1))
        return F.log_softmax(prediction, dim=1)