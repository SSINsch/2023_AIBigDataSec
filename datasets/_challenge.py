import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from typing import Tuple, List
from utils import prepare_sequence

logger = logging.getLogger(__name__)


def load_challengeset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 중복 데이터 제거
    df = df.drop_duplicates(subset=['payload', 'label_action'])

    # df 에서 index들 찾아서 빼기
    index = [49, 294, 649, 747, 1026, 2035, 2051, 2297, 3152, 3326, 3863, 5174, 5579,
             6693, 7160, 8002, 8469, 8470, 9243, 9271, 9294, 9402, 10325, 10963, 12009,
             12595, 12643, 15499, 15905, 16045, 16294, 16417, 16470, 17531, 17716, 18044,
             18440, 18555, 18827, 19068, 19155, 19488, 20167, 20183, 21213, 21298, 22632,
             22672, 23120, 23480, 23890, 24936, 25128, 25378, 25803, 27310, 27718, 28415,
             29692, 30139, 31157, 32592, 33309, 33750, 33903, 33943, 35989, 36032, 36280,
             36890, 36984, 37191, 37329, 37495, 38034, 39549, 39605, 39655, 41249, 41935,
             42565, 42827]
    df = df[~df['Log_Number'].isin(index)]

    return df


class ChallengeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer,
                 label_to_num,
                 tok2idx: List[int] = None):
        self.data = load_challengeset(data_path)
        self.log_num = self.data['Log_Number']
        self.payload = self.data['payload']
        self.label_to_num = label_to_num
        self.target = self._preprocess_label()
        self.tokenizer = tokenizer
        self.tok2idx = tok2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x_payload = self.payload.iloc[index]
        i_log_num = self.log_num[index]
        y_action = self.target[index]
        x_payload_tokenized = self.tokenizer.tokenize(x_payload)

        # get index of words(token) and characters(of each word)
        x_idx_tokens_item = prepare_sequence(x_payload_tokenized, self.tok2idx)

        return x_payload_tokenized, x_idx_tokens_item, y_action, i_log_num

    def _preprocess_label(self):
        # label => str => number 변경
        y = self.data['label_action'].tolist()
        y_num = []
        for lb in y:
            y_num.append(self.label_to_num[lb])

        return y_num


class BasicCollator:
    def __init__(self, block_size: int):
        self.block_size = block_size

    def __call__(self, data: Tuple[List, List]):
        # do something with batch and self.params
        data.sort(key=lambda x: len(x[0]), reverse=True)

        x_payload_tokenized, x_idx_tokens_item, y_action, index_log_num = zip(*data)

        # label -> Torch
        y_action_torched = torch.tensor(y_action)

        # index data -> Torch
        # get max length of input
        x_lens = [len(x) for x in x_idx_tokens_item]
        max_x_len = max(x_lens)
        if max_x_len < self.block_size:
            _block_size = max_x_len
        else:
            _block_size = self.block_size

        # padding & slicing
        blocked_x_idx_tokens_item = []
        for i, x in enumerate(x_idx_tokens_item):
            if len(x) < _block_size:
                padding_list = [0] * (_block_size - len(x))
                blocked_x_idx_tokens_item.append(x + padding_list)
            else:
                blocked_x_idx_tokens_item.append(x[:_block_size])

        padded_blocked_x_idx_tokens_matrix = torch.tensor(blocked_x_idx_tokens_item)

        return x_payload_tokenized, padded_blocked_x_idx_tokens_matrix, y_action_torched, index_log_num


def get_loader(data_path: str,
               tokenizer,
               label_to_num,
               block_size: int,
               tok2idx=None,
               collate_method: str = 'collate_basic',
               batch_size: int = 256,
               shuffle: bool = True):
    collate = None
    if collate_method == 'collate_basic':
        collate = BasicCollator(block_size=block_size)
    else:
        logger.warning(f'Unknown collate method: {collate_method}')

    logger.debug(f'load dataset with {data_path}')
    dataset = ChallengeDataset(data_path, tokenizer, tok2idx=tok2idx, label_to_num=label_to_num)

    logger.debug('load dataloader')
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=collate)
    return data_loader
