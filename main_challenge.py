from utils import build_tok_vocab
from datasets import load_challengeset
from tokenizers import LogTokenizer
from models import TextCNN, BiLSTM
from datasets import get_loader
from trainer import ChallengeTrainer

import random
import torch
import logging.config
import json
import torch.nn as nn
from gensim.models import Word2Vec
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

config = json.load(open('./logger.json'))
logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

SEED = 0
HIDDEN_DIM = 300
NUM_LSTM_LAYER = 2
n_classes = 9
EMBED_SIZE = 200
learning_rate = 10e-3
BATCH_SIZE = 512
block_size = 128
output_dir = r'./model_save'
num_epoch = 50
PRE_EMBED = None
MAX_VOCAB = 29998
path_train = r'./data/train.csv'
path_challenge = r'./data/challenge.csv'


if __name__ == '__main__':
    # 시드 고정
    random.seed(SEED)
    torch.manual_seed(SEED)
    logger.debug(f'seed: {SEED}')

    # CUDA setting 확인
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    logger.debug(f'device: {DEVICE}')

    # vocab set
    # load dataset but except some items (duplicate, multi label)
    data = load_challengeset(path_train)
    payloads = data['payload']
    tokenizer = LogTokenizer()
    func_tok_vocab_set, func_tok2idx = build_tok_vocab(payloads, tokenizer, min_freq=1, max_vocab=MAX_VOCAB)
    logger.info(f'Vocab set size: {len(func_tok2idx)}')

    # 문자열 라벨을 숫자로 바꿔주는 변수
    lbstr_to_lbnum = list(dict(data['label_action'].value_counts()).keys())
    lbstr_to_lbnum = {label: i for i, label in enumerate(lbstr_to_lbnum)}

    # model
    if PRE_EMBED is not None:
        embed_model = Word2Vec.load(PRE_EMBED)
        embed_model = torch.FloatTensor(embed_model.wv.vectors)
    else:
        embed_model = None
    bilstm_model = BiLSTM(hidden_dim=HIDDEN_DIM,
                          num_lstm_layer=NUM_LSTM_LAYER,
                          n_classes=n_classes,
                          embed_size=EMBED_SIZE,
                          vocab_size=len(func_tok2idx),
                          pre_embedding=embed_model)
    logger.info(bilstm_model)
    bilstm_model = bilstm_model.to(DEVICE)
    optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # load data
    challenge_loader = get_loader(data_path=path_challenge,
                                  batch_size=BATCH_SIZE,
                                  tokenizer=tokenizer,
                                  tok2idx=func_tok2idx,
                                  block_size=block_size,
                                  label_to_num=lbstr_to_lbnum)

    # trainer
    # train_loader에 challegne loader를 넣긴 했지만, 사용을 안 하므로 걱정X
    challe_trainer = ChallengeTrainer(model=bilstm_model,
                                      optimizer=optimizer,
                                      criterion=criterion,
                                      device=DEVICE,
                                      output_dir=output_dir,
                                      train_loader=challenge_loader)

    # challenge() 함수에서 필수적으로 모델을 load하도록 구성되어 있음
    logger.info('CHALLENGE START')
    answer_df = challe_trainer.challenge(model_summary_path=r'./model_save/BiLSTM_train_253.bin',
                                         challenge_loader=challenge_loader)

    # 정답 출력하기
    # 일단 label number로 출력
    logger.info('Save csv answer file')
    answer_df.to_csv('answer_num.csv', index=False)

    # label_action 치환해서 str로 출력
    lbnum_to_lbstr = {lbstr_to_lbnum[k]:k for k in lbstr_to_lbnum.keys()}
    answer_df = answer_df.replace({"label_action": lbnum_to_lbstr})
    answer_df.to_csv('answer_str.csv', index=False)
