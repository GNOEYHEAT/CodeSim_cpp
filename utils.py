import numpy as np
import random
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

def set_args():
    parser = ArgumentParser(description="preprocess")
    parser.add_argument('--text_pretrained_model', type=str, required=True, choices=['microsoft/codebert-base', 'microsoft/graphcodebert-base', 'microsoft/unixcoder-base'])
    parser.add_argument('--truncation_side', type=str, required=False, default='left') # right or left
    parser.add_argument('--bm25', type=str, required=False, default='bm25plus')
    parser.add_argument('--frac', type=float, required=False, default=0.01 )
    parser.add_argument('--seed', type=int, required=False, default=826)
    parser.add_argument('--device', type=int, required=False, default=0)
    parser.add_argument('--num_workers', type=int, required=False, default=0)
    args = parser.parse_args('')
    return args

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)