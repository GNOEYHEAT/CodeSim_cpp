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

| CodeBERT Model          | Private Accuracy | Public Accuracy | Val Accuracy |
|-------------------------|------------------|-----------------|--------------|
| GraphCodeBERT           | -                | 0.9889212828    | -            |
| UniXcoder               | -                | 0.9891133596    | -            |
| GraphCodeBERT+UniXcoder | 0.9911           | 0.9908420511    | -            |
