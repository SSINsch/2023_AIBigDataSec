from utils import build_tok_vocab
from datasets import load_challengeset
from tokenizers import LogTokenizer
from models import BiLSTM
from datasets import get_loader
from trainer import ChallengeTrainer

import datetime
import random
import torch
import logging.config
import json
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from gensim.models import Word2Vec
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

config = json.load(open('./logger.json'))
logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

SEED = 0
path_train = r'./data/train.csv'
path_valid = r'./data/valid.csv'
path_test = r'./data/test.csv'
HIDDEN_DIM = 300
NUM_LSTM_LAYER = 2
n_classes = 9
EMBED_SIZE = 200
learning_rate = 10e-3
BATCH_SIZE = 256
block_size = 128
output_dir = r'./model_save'
num_epoch = 50
PRE_EMBED = None
MAX_VOCAB = 29998

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
    func_tok_vocab_set, func_tok2idx = build_tok_vocab(payloads, tokenizer, min_freq=1)
    logger.info(f'Vocab set size: {len(func_tok2idx)}')

    # 문자열 라벨을 숫자로 바꿔주는 변수
    lbstr_to_lbnum = list(dict(data['label_action'].value_counts()).keys())
    lbstr_to_lbnum = {label: i for i, label in enumerate(lbstr_to_lbnum)}

    # model
    if PRE_EMBED is not None:
        # wv_model = KeyedVectors.load_word2vec_format(PRE_EMBED)
        # wv_model = wv_model.vectors
        embed_model = Word2Vec.load(PRE_EMBED)
        embed_model = torch.FloatTensor(embed_model.wv.vectors)
    else:
        embed_model = None

    # model
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
    train_loader = get_loader(data_path=path_train,
                              batch_size=BATCH_SIZE,
                              tokenizer=tokenizer,
                              tok2idx=func_tok2idx,
                              block_size=block_size,
                              label_to_num=lbstr_to_lbnum)

    eval_loader = get_loader(data_path=path_valid,
                             batch_size=BATCH_SIZE,
                             tokenizer=tokenizer,
                             tok2idx=func_tok2idx,
                             block_size=block_size,
                             label_to_num=lbstr_to_lbnum)

    test_loader = get_loader(data_path=path_test,
                             batch_size=BATCH_SIZE,
                             tokenizer=tokenizer,
                             tok2idx=func_tok2idx,
                             block_size=block_size,
                             label_to_num=lbstr_to_lbnum)

    challe_trainer = ChallengeTrainer(model=bilstm_model,
                                      optimizer=optimizer,
                                      criterion=criterion,
                                      device=DEVICE,
                                      output_dir=output_dir,
                                      train_loader=train_loader,
                                      eval_loader=eval_loader,
                                      test_loader=test_loader)

    now = datetime.datetime.now().strftime('%Y-%m-%d')
    writer = SummaryWriter(f'logs/{bilstm_model.name}/{now}')

    # train / eval
    for epoch in range(num_epoch):
        logger.info(f'EPOCH: [ {epoch + 1}/{num_epoch} ]')
        train_result, model_summary_path = challe_trainer.train(n_epoch=epoch+1)
        eval_result, _ = challe_trainer.evaluate(mode='eval', n_epoch=epoch+1, model_summary_path=None)
        test_result, _ = challe_trainer.evaluate(mode='test', n_epoch=epoch+1, model_summary_path=None)

        writer.add_scalars(main_tag='Loss/train_eval',
                           global_step=epoch,
                           tag_scalar_dict={'train_loss': train_result['Avg loss'],
                                            'val_loss': eval_result['Avg loss']})
        writer.add_scalars(main_tag='Acc/train_eval_test',
                           global_step=epoch,
                           tag_scalar_dict={'train_acc': train_result['Avg acc'],
                                            'val_acc': eval_result['Acc'],
                                            'test_acc': test_result['Acc']})
        writer.add_scalars(main_tag='F1/train_eval_test',
                           global_step=epoch,
                           tag_scalar_dict={'train_f1': train_result['Avg f1'],
                                            'val_f1': eval_result['F1'],
                                            'test_f1': test_result['F1']})

    writer.close()
