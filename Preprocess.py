import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # %matplotlib inline
import seaborn as sns

from argparse import ArgumentParser
import re
from glob import glob
from tqdm import tqdm
from itertools import combinations
from utils import set_args, set_seeds

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import transformers
from transformers import AutoTokenizer, AutoModel
transformers.logging.set_verbosity_error()

from rank_bm25 import BM25Okapi, BM25L, BM25Plus
# from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')


def code_cpp_preprocessing():

    ''' Args Set '''
    args = set_args()
    ''' Seed Set '''
    set_seeds(args.seed)

    idx = f"{args.text_pretrained_model}_{args.bm25}"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    '''
    Embedding Size
    graphcodebert-base" == 512
    codebert-base" == 512
    unixcoder-base == 1024
    '''
    tokenizer = AutoTokenizer.from_pretrained(args.text_pretrained_model)
    tokenizer.truncation_side = args.truncation_side
    print('1')

if __name__ == "__main__":
    code_cpp_preprocessing()

