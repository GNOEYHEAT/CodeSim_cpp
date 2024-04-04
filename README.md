# CodeSim_cpp

The following codes are the solutions **(1st place, private score: 0.9911)** for the dacon competition.

## 1. Environmental settings
### 1.1 Clone this repository 

```bash
git clone https://github.com/GNOEYHEAT/CodeSim_cpp.git
cd CodeSim_cpp
```

### 1.2 Install packages
```bash
pip install -r requirements.txt 
```

## Directory Structure

```bash
├── Dataset
│   ├── train_code
│   │   ├── problem001
│   │   ├── ...
│   │   └── problem500
│   ├── sample_submission.csv
│   ├── sample_train.csv
│   └── test.csv
├── Utils
│   ├── CodeLM_utils.py
│   └── Preprocessing_utils.py
├── Preprocess.py
└── CodeLM.py
```
## 2. Data Proprocessing

```bash
python Preprocess.py 
```

## 3. Train Model

```bash
python CodeLM.py
```


## 4. Experiment Results

The final submission is **GraphCodeBERT+UniXcoder**.

### Preprocessed Datasets
[Datasets](https://huggingface.co/datasets/GNOEYHEAT/CodeSim_cpp)

* The hyperparameters are as follows:
    - truncation_side='left', bm25='bm25plus'

### Pre-trained Models
[Models](https://huggingface.co/GNOEYHEAT/CodeSim_cpp/tree/main/models)

* The hyperparameters are as follows:
    - truncation_side='left', optimizer='adamw', learning_rate=0.00003

### Results
[Preds](https://huggingface.co/GNOEYHEAT/CodeSim_cpp/tree/main/preds)

| index     | CodeBERT Model          | frac | text_len | Pr Acc  | Pl Acc  | Val Acc |
|-----------|-------------------------|------|----------|---------|---------|---------|
| exp_30    | GraphCodeBERT           | 0.01 | 512      | 0.98859 | 0.98831 | 0.99641 |
| exp_31    | GraphCodeBERT           | 0.02 | 512      | 0.98909 | 0.98892 | 0.99794 |
| exp_32    | UniXcoder               | 0.01 | 1024     | 0.98942 | 0.98911 | 0.99606 |
| exp_31+32 | GraphCodeBERT+UniXcoder | -    | -        | 0.99111 | 0.99084 | -       |
