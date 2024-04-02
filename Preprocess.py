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
from Utils.Preprocessing_utils import *

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
    args = set_pp_args()

    ''' Seed Set '''
    set_seeds(args.seed)

    ''' Make Preprocessed Data Directory '''
    pp_mkdir(args)
    
    ''' CUDA'''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ''' Tokenizer '''
    tokenizer = AutoTokenizer.from_pretrained(args.text_pretrained_model)
    tokenizer.truncation_side = args.truncation_side

    ''' Train Data Preprocessing and Generation '''
    train_df = create_df(args)
    train_df_bm25 = get_pairs(train_df, tokenizer, args)
    f_split(train_df_bm25, args)

    '''Test Code preprocessing '''
    test_code_df(args)

if __name__ == "__main__":
    code_cpp_preprocessing()

