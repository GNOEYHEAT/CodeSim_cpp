import pandas as pd
import numpy as np
import random
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from glob import glob
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from tqdm import tqdm
from itertools import combinations
import re
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

def set_CodeLM_args():
    parser = ArgumentParser(description="CodeLM")
    parser.add_argument('--text_pretrained_model', default="microsoft/graphcodebert-base", type=str) # microsoft/codebert-base, microsoft/graphcodebert-base, microsoft/unixcoder-base
    parser.add_argument('--text_len', default=512, type=int)
    parser.add_argument('--truncation_side', default='left', type=str) # right or left
    parser.add_argument('--optimizer', default="adamw", type=str)
    parser.add_argument('--learning_rate', default=0.00003, type=float)
    parser.add_argument('--scheduler', default="none", type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--cv', default=5, type=int)
    parser.add_argument('--seed', default=826, type=int)
    parser.add_argument('--mixed_precision', default=16, type=int)
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--latent_dim', default=768, type=int)
    args = parser.parse_args('')
    return args

def load_data(): 
    train_df = pd.read_csv("Dataset/sample_train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    return train_df, test_df



class TextDataset(Dataset):
    def __init__(self, df, tokenizer, args, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.args = args
        self.is_test = is_test
        if args.truncation_side == "left":
            self.tokenizer.truncation_side = 'left'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            row['code1'], row['code2'],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.args.text_len,
            return_tensors="pt",
        )

        for k,v in encoding.items():
            encoding[k] = v.squeeze()

        if not self.is_test:
            labels = torch.tensor(row["similar"], dtype=torch.float)
            return encoding, labels

        return encoding
    
class TextModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.txt_model = AutoModel.from_pretrained(args.text_pretrained_model)
        self.classifier = nn.Sequential(
            nn.Linear(args.latent_dim, 1)
        )

    def forward(self, inputs):
        txt_side = self.txt_model(**inputs)
        txt_feature = txt_side.last_hidden_state[:, 0, :]
        outputs = self.classifier(txt_feature)
        return outputs

class TextClassifier(pl.LightningModule):
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone
        self.args = args
    def forward(self, x):
        outputs = self.backbone(x)
        return outputs

    def step(self, batch):
        x = batch[0]
        y = batch[1]
        y_hat = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(y_hat.squeeze(), y)
        return loss, y, y_hat

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        y_pred = (y_hat > 0).float().squeeze()
        acc = (y_pred == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        y_pred = (y_hat > 0).float().squeeze()
        acc = (y_pred == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self.forward(batch)
        return y_hat

    def configure_optimizers(self):
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9)
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        if self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        
        if self.args.scheduler == "none":
            return optimizer
        if self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args.epochs//2,
                eta_min=self.args.learning_rate//10,
            )
            return [optimizer], [scheduler]
        
def saved_mkdir():

     if not os.path.exists(f"./saved"):
        os.makedirs(f"./saved/")