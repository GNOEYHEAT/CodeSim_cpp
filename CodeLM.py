import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from Utils.Preprocessing_utils import *
from Utils.CodeLM_utils import *

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
transformers.logging.set_verbosity_error()


import re
from tqdm import tqdm
from glob import glob
from itertools import combinations

import warnings
warnings.filterwarnings('ignore')
from argparse import ArgumentParser


def CodeLM():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''' Set Args'''
    args = set_CodeLM_args()
    idx = f"{args.text_pretrained_model}".replace('/', '_')

    ''' Set Seed'''
    set_seeds(args.seed)

    ''' Set Tokenizer and LM Model'''
    tokenizer = AutoTokenizer.from_pretrained(args.text_pretrained_model)
    model = AutoModel.from_pretrained(args.text_pretrained_model)

    ''' Make saved Directory '''
    saved_mkdir()

    ''' Load Data '''
    df, test_df = load_data()

    ''' Model Train Pipline '''
    val_acc_list = []
    preds_list = []

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    for i, (train_index, val_index) in enumerate(skf.split(df, df['similar'])):

        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]    

        ''' DataLoader '''
        train_ds = TextDataset(train_df, tokenizer, args, False)
        val_ds = TextDataset(val_df, tokenizer, args, False)
        test_ds = TextDataset(test_df, tokenizer, args, True)
        train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        ''' Model '''
        model = TextClassifier(TextModel(args), args)

        ''' Set Model Checkpoint'''
        callbacks = [
            # pl.callbacks.EarlyStopping(
            #     monitor="val_acc", patience=3, mode="max"
            # ),
            pl.callbacks.ModelCheckpoint(
                dirpath="saved/", filename=f"{idx}_{i}",
                monitor="val_acc", mode="max"
            ),
        ]

        ''' Set Trainer '''
        trainer = pl.Trainer(
            max_epochs=args.epochs, accelerator="auto", callbacks=callbacks,
            precision=args.mixed_precision, #logger=wandb_logger,
            devices=args.device, #strategy='ddp_find_unused_parameters_true'
        )

        ''' Fit '''
        trainer.fit(model, train_dataloader, val_dataloader)

        ''' Model Load '''
        ckpt = torch.load(f"saved/{idx}_{i}.ckpt")
        model.load_state_dict(ckpt['state_dict'])

        ''' Calculate Validation '''
        eval_dict = trainer.validate(model, dataloaders=val_dataloader)[0]
        val_acc_list.append(eval_dict["val_acc"])
        
        ''' Predict '''
        preds = trainer.predict(model, dataloaders=test_dataloader)
        preds_list.append(np.vstack(preds))

        break

    ''' Predict '''
    preds = np.mean(preds_list, axis=0)

    ''' Submission file '''
    submission = pd.read_csv('data/sample_submission.csv')
    submission['similar'] = np.where(preds>0, 1, 0)
    submission.to_csv(f'{idx}.csv', index=False)

if __name__ == "__main__":
    CodeLM()