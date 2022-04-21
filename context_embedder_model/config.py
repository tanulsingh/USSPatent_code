import torch
from transformers import AutoTokenizer

main_dir = '/home/tanulsingh/USSPatent_code'
version = 'v0'
SAVE_DIR = f'model/{version}'

NUM_FOLDS = 5
NUM_WORKERS = 4
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 64
EPOCHS = 5
SEED = 41
warmup_ratio = 0
LR = 2e-5
WEIGHT_DECAY = 0.01
mixed_precision = True
pooling = 'attention'

device = torch.device('cuda')

transformer_model = 'anferico/bert-for-patents'
TOKENIZER = AutoTokenizer.from_pretrained(transformer_model)
MAX_LENGTH = 32

SCHEDULER = 'linear'