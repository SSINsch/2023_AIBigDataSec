import argparse
import logging
import collections
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    return args


def build_tok_vocab(tokenize_target: List,
                    tokenizer,
                    min_freq: int = 3,
                    max_vocab=29998) -> Tuple[List[str], Dict]:

    vocab = []
    logger.debug('start parsing vocabulary set!')
    for i, target in enumerate(tokenize_target):
        try:
            temp = tokenizer.tokenize(target)
            vocab.extend(temp)

        except Exception as e_msg:
            error_target = f'idx: {i} \t target:{target}'
            logger.warning(error_target, e_msg)

    logger.debug('vocabulary set parsing done')
    logger.debug('start configuring vocabulary set!')
    vocab = collections.Counter(vocab)
    temp = {}
    # min_freq보다 적은 단어 거르기
    for key in vocab.keys():
        if vocab[key] >= min_freq:
            temp[key] = vocab[key]
    vocab = temp

    # 가장 많이 등장하는 순으로 정렬한 후, 적게 나온것 위주로 vocab set에서 빼기
    vocab = sorted(vocab, key=lambda x: -vocab[x])
    if len(vocab) > max_vocab:
        vocab = vocab[:max_vocab]

    tok2idx = {'<pad>': 0, '<unk>': 1}
    for tok in vocab:
        tok2idx[tok] = len(tok2idx)
    vocab.extend(['<pad>', '<unk>'])

    logger.debug('vocabulary set configuring done')

    return vocab, tok2idx


def prepare_sequence(seq, word_to_idx):
    indices = []

    for word in seq:
        if word not in word_to_idx.keys():
            indices.append(word_to_idx['<unk>'])
        else:
            indices.append(word_to_idx[word])

    # indices = np.array(indices)

    return indices


def to_np(x):
    return x.detach().cpu().numpy()
