import torch
from transformers import AutoTokenizer

version = 'v0'
SAVE_DIR = f'model/{version}'

context_type = 'a'
solution_type = 'STScrossenc'

NUM_WORKERS = 4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32
EPOCHS = 2
SEED = 81
warmup_ratio = 0.05
LR = 2e-5
WEIGHT_DECAY = 0.01
mixed_precision=True
pooling = 'mean'

device = torch.device('cuda')

transformer_model = 'anferico/bert-for-patents'
TOKENIZER = AutoTokenizer.from_pretrained(transformer_model)
MAX_LENGTH = 64

SCHEDULER = 'linear'