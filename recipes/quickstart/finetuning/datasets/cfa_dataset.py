# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
from datasets import load_dataset
import itertools
import torch

def get_custom_dataset(dataset_config, tokenizer, split, split_ratio=0.9):
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset("ChanceFocus/flare-cfa")
    dataset = dataset_dict['test']
    print(dataset.column_names)

    dataset = dataset.train_test_split(test_size=1-split_ratio, shuffle=True, seed=42)[split]

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["query"], add_special_tokens=False)
        answer = tokenizer.encode(sample["answer"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
