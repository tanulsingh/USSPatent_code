import random
import os
import numpy as np
import torch
from scipy import stats 
import config

import torch.nn as nn

from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PearsonMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = 0
        self.y_pred = 1
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))

        self.score = stats.pearsonr(self.y_true, self.y_pred)[0]
    
    @property
    def avg(self):
        return self.score


def fetch_scheduler(optimizer,warm_up,total_steps=None):
    if config.SCHEDULER == 'linear':
        scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_up,
        num_training_steps=total_steps,
        )
    
    elif config.SCHEDULER == 'cos':
        scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_up,
        num_training_steps=total_steps,
        )
    
    elif config.SCHEDULER == 'constant':
        scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_up
        )

    return scheduler


def get_logger(filename=config.SAVE_DIR+'/train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
