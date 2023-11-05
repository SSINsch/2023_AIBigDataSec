from datasets import load_challengeset
from tokenizers import LogTokenizer
import random
import torch
import logging.config
import json
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
EMBED_SIZE = 200
EPOCH = 20
output_dir = r'./model_save'

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

    sentences = []
    for payload in payloads:
        s = tokenizer.tokenize(payload)
        sentences.append(s)
    sentences.append(['<pad>', '<unk>'])

    embed_model = Word2Vec(sentences=sentences,
                           vector_size=EMBED_SIZE,
                           window=5,
                           min_count=1,
                           workers=4,
                           seed=SEED,
                           epochs=EPOCH,
                           sg=0)

    # 모델 저장
    model_name = f'word2vec_LogTokenizer_epoch{EPOCH}_win5_count3_cbow_seed{SEED}.bin'
    # embed_model.wv.save_word2vec_format(f'{output_dir}/{model_name}')
    embed_model.save(f'{output_dir}/{model_name}')

    # model = KeyedVectors.load_word2vec_format(f'{output_dir}/{model_name}')
    model = Word2Vec.load(f'{output_dir}/{model_name}')
    print(model)

