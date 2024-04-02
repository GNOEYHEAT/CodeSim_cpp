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

def set_args():
    parser = ArgumentParser(description="preprocess")
    parser.add_argument('--text_pretrained_model', type=str, default='microsoft/graphcodebert-base')
    parser.add_argument('--truncation_side', type=str, default='left') # right or left
    parser.add_argument('--bm25', type=str, default='bm25plus')
    parser.add_argument('--frac', type=float, default=0.01 )
    parser.add_argument('--seed', type=int,  default=826)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--code_path', type=str, default="Dataset/train_code")
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

def read_cpp_code():
    train_code_paths = glob('Dataset/train_code/*/*.cpp') # Data path
    train_code_paths = train_code_paths[0]
    with open(train_code_paths, 'r', encoding='utf-8') as file:
        return file.read()
    
def clean_data(script, data_type="dir"):
    if data_type == "dir":
        with open(script, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            preproc_lines = []
            in_multiline_comment = False
            for line in lines:
                if line.startswith('#include'): # #include로 시작하는 행 제거
                    continue
                line = line.strip().replace('\t', '').split('//')[0].strip() # 개행문자 제거, 주석 제거
                line = re.sub(' +', ' ', line) # 개행문자 제거
                if line == '': # 전처리 후 빈 라인은 skip
                    continue
                # 여러 줄 주석 시작
                if '/*' in line:
                    in_multiline_comment = True
                # 여러 줄 주석 안에 있는 내용은 무시
                if not in_multiline_comment:
                    preproc_lines.append(line)
                # 여러 줄 주석 종료
                if '*/' in line:
                    in_multiline_comment = False

    elif data_type == "file":
        lines = script.split('\n')
        preproc_lines = []
        in_multiline_comment = False
        for line in lines:
            if line.startswith('#include'): # #include로 시작하는 행 제거
                continue
            line = line.strip().replace('\t', '').split('//')[0].strip() # 개행문자 제거, 주석 제거
            line = re.sub(' +', ' ', line) # 개행문자 제거
            if line == '': # 전처리 후 빈 라인은 skip
                continue
            # 여러 줄 주석 시작
            if '/*' in line:
                in_multiline_comment = True
            # 여러 줄 주석 안에 있는 내용은 무시
            if not in_multiline_comment:
                preproc_lines.append(line)
            # 여러 줄 주석 종료
            if '*/' in line:
                in_multiline_comment = False

    processed_script = ' '.join(preproc_lines) # 개행 문자로 합침
    # processed_script = '\n'.join(preproc_lines) # 개행 문자로 합침
    return processed_script

''' Create positive and negative pair  '''
def get_pairs(input_df, tokenizer, args):
    codes = input_df['code'].to_list()
    problems = input_df['problem_num'].unique().tolist()
    problems.sort()

    tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
    if args.bm25 == "bm25ok":
        bm25 = BM25Okapi(tokenized_corpus)
    if args.bm25 == "bm25l":
        bm25 = BM25L(tokenized_corpus)
    if args.bm25 == "bm25plus":
        bm25 = BM25Plus(tokenized_corpus)

    total_positive_pairs = []
    total_negative_pairs = []

    for problem in tqdm(problems[:1]):
        solution_codes = input_df[input_df['problem_num'] == problem]['code']
        positive_pairs = list(combinations(solution_codes.to_list(),2))

        solution_codes_indices = solution_codes.index.to_list()
        negative_pairs = []

        first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
        negative_code_scores = bm25.get_scores(first_tokenized_code)
        negative_code_ranking = negative_code_scores.argsort()[::-1] # 내림차순
        ranking_idx = 0

        for solution_code in solution_codes:
            negative_solutions = []
            while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
                high_score_idx = negative_code_ranking[ranking_idx]

                if high_score_idx not in solution_codes_indices:
                    negative_solutions.append(input_df['code'].iloc[high_score_idx])
                ranking_idx += 1

            for negative_solution in negative_solutions:
                negative_pairs.append((solution_code, negative_solution))

        total_positive_pairs.extend(positive_pairs)
        total_negative_pairs.extend(negative_pairs)

    pos_code1 = list(map(lambda x:x[0],total_positive_pairs))
    pos_code2 = list(map(lambda x:x[1],total_positive_pairs))

    neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
    neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

    pos_label = [1]*len(pos_code1)
    neg_label = [0]*len(neg_code1)

    pos_code1.extend(neg_code1)
    total_code1 = pos_code1
    pos_code2.extend(neg_code2)
    total_code2 = pos_code2
    pos_label.extend(neg_label)
    total_label = pos_label
    
    pair_data = pd.DataFrame(data={
        'code1':total_code1,
        'code2':total_code2,
        'similar':total_label
    })
    
    return pair_data

def create_df(args):

    code_folder = args.code_path
    problem_folders = os.listdir(args.code_path)
    processed_scripts = []
    problem_nums = []

    for problem_folder in tqdm(problem_folders):
        scripts = os.listdir(os.path.join(code_folder, problem_folder))
        problem_num = scripts[0].split('_')[0]
        for script in scripts:
            script_file = os.path.join(code_folder, problem_folder, script)
            processed_script = clean_data(script_file, data_type="dir")
            processed_scripts.append(processed_script)
        problem_nums.extend([problem_num] * len(scripts))
        
    pp_train_df = pd.DataFrame(
        data={'code': processed_scripts, 'problem_num': problem_nums})
    
    return pp_train_df 

def f_split(df, args):
    plength = len(df) // 10
    model_name = args.text_pretrained_model
    model_name = model_name.replace('/','_')
    idx = f"{model_name}_{args.bm25}"

    for i in tqdm(range(10)):
        temp_df = df.iloc[i*plength:(i+1)*plength]
        temp_df.to_parquet(f'./Dataset/pp_train_{model_name}/pp_train_{idx}_{i}.parquet', engine='pyarrow', index=False)

def test_code_df(args):
    test_df = pd.read_csv("Dataset/test.csv")
    code1 = test_df['code1'].values
    code2 = test_df['code2'].values
    processed_code1 = []
    processed_code2 = []

    for i in tqdm(range(len(code1))):
        processed_c1 = clean_data(code1[i], data_type="file")
        processed_c2 = clean_data(code2[i], data_type="file")
        processed_code1.append(processed_c1)
        processed_code2.append(processed_c2)

    model_name = args.text_pretrained_model
    model_name = model_name.replace('/','_')
    idx = f"{model_name}_{args.bm25}"

    pp_test_df = pd.DataFrame(
        list(zip(processed_code1, processed_code2)), columns=["code1", "code2"])
    pp_test_df.to_parquet(f'./Dataset/pp_train_{model_name}/pp_test_{idx}.parquet', engine='pyarrow', index=False)

def pp_mkdir(args):
     model_name = args.text_pretrained_model
     model_name = model_name.replace('/','_')

     if not os.path.exists(f"./Dataset/pp_train_{model_name}"):
        os.makedirs(f"./Dataset/pp_train_{model_name}")
         
     if not os.path.exists(f"./Dataset/pp_test_{model_name}"):
        os.makedirs(f"./Dataset/pp_test_{model_name}")