import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class TextCNN(nn.Module):
    name = 'TextCNN'

    def __init__(self, n_classes,
                 pre_embedding=None,
                 vocab_size=10000,
                 embed_size=200,
                 kernel_windows=[3, 4, 5],
                 input_channel=1,
                 dropout=0.25):
        super(TextCNN, self).__init__()

        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pre_embedding = pre_embedding
        if pre_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(self.pre_embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embed_size)

        self.kernel_windows = kernel_windows
        self.input_channel = input_channel
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.input_channel, self.embed_size, kernel_size=(k, self.embed_size)) for k in self.kernel_windows]
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(len(self.kernel_windows) * self.embed_size, self.n_classes)

    def forward(self, x):
        # input (128, 1000) : (128 batch size), (1000 아마 word token 개수??)
        embed = self.embedding(x)
        # => embed (128, 1000, 300) : (128 batch size), (1000 아마 word token 개수??), (300 임베딩 차원)
        embed = embed.unsqueeze(1)
        # => embed (128, 1, 1000, 300)
        conv_x = [conv(embed) for conv in self.convs]
        # => convs_x size = [(128, 300, 998, 1), [(128, 300, 997, 1), [(128, 300, 996, 1)]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in conv_x]
        # => pool_x size = [(128, 300, 1), [(128, 300, 1), [(128, 300, 1)]
        linear_x = torch.cat(pool_x, dim=1)
        # => linear_x size = (128, 900, 1)
        linear_x = linear_x.squeeze(-1)
        # => linear_x size = (128, 900)
        linear_drop_x = self.dropout(linear_x)
        logit = self.linear(linear_drop_x)
        # logit = F.softmax(logit, dim=1)

        return logit
