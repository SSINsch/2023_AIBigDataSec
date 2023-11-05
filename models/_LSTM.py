import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class BiLSTM(nn.Module):
    name = 'BiLSTM'

    def __init__(self, hidden_dim, num_lstm_layer, n_classes,
                 pre_embedding=None,
                 vocab_size=10000,
                 embed_size=200,
                 dropout=0.1):
        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_lstm_layer = num_lstm_layer
        self.n_classes = n_classes
        self.dropout = dropout
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.pre_embedding = pre_embedding
        if pre_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(self.pre_embedding)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embed_size)

        # BiLSTM layer 세팅
        self.bi_lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_lstm_layer,
                               dropout=self.dropout,
                               batch_first=True,
                               bidirectional=True)

        # bidirectional 이라서 hidden_dim * 2
        self.linear = nn.Linear(self.hidden_dim * 2, self.n_classes)

    def forward(self, x):
        # embedding
        embedded = self.embedding(x)

        # lstm 통과
        lstm_out, (h_n, c_n) = self.bi_lstm(embedded)  # (h_0, c_0) = (0, 0)

        # forward와 backward의 마지막 time-step의 은닉 상태를 가지고 와서 concat
        # 이때 모델이 batch_first라는 점에 주의한다. (dimension 순서가 바뀜)
        hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        out = self.linear(hidden)

        return out
