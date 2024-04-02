# CodeSim_cpp

The following codes are the solutions **(1st place, private score: 0.9911)** for the dacon competition.

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

## Experiments

The final submission is **GraphCodeBERT+UniXcoder**.

https://huggingface.co/GNOEYHEAT/CodeSim_cpp

# CodeSim_cpp

The following codes are the solutions **(1st place, private score: 0.9911)** for the dacon competition.

## Experiments

The final submission is **GraphCodeBERT+UniXcoder**.

https://huggingface.co/GNOEYHEAT/CodeSim_cpp

truncation_side='left', bm25='bm25plus', seed=826

| index     | CodeBERT Model          | frac | Pr Acc  | Pl Acc  | Val Acc |
|-----------|-------------------------|------|---------|---------|---------|
| exp_30    | GraphCodeBERT           | 0.01 | 0.98859 | 0.98831 | 0.99641 |
| exp_31    | GraphCodeBERT           | 0.02 | 0.98909 | 0.98892 | 0.99794 |
| exp_32    | UniXcoder               | 0.01 | 0.98942 | 0.98911 | 0.99606 |
| exp_31+32 | GraphCodeBERT+UniXcoder | -    | 0.99111 | 0.99084 | -       |
