# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
from datasets import load_dataset
import itertools
import torch

def get_custom_dataset(dataset_config, tokenizer, split, split_ratio=0.9):
    dataset_dict = load_dataset("ChanceFocus/flare-cfa")
    dataset = dataset_dict['test']
    print(dataset.column_names)

    dataset = dataset.train_test_split(test_size=1-split_ratio, shuffle=True, seed=42)[split]

    def tokenize_add_label(sample, dataset, num_examples=3):
        examples = dataset.select(range(num_examples))
        full_prompt = tokenizer.bos_token
        
        for ex in examples:
            if full_prompt == tokenizer.bos_token:
                full_prompt +=  ex["query"] + ex["answer"] + "\n"
            else:
                full_prompt +=  ex["text"] + " Answer:" + ex["answer"] + "\n"
        
        if full_prompt == tokenizer.bos_token:
            full_prompt += sample["query"]
        else:
            full_prompt += sample["text"] + " Answer:"
        
        prompt = tokenizer.encode(full_prompt, add_special_tokens=False)
        answer = tokenizer.encode(sample["answer"] + tokenizer.eos_token, add_special_tokens=False)
        
        # print("Full prompt + answer:", full_prompt + sample["answer"] + tokenizer.eos_token)
        
        sample = {
            "input_ids": prompt + answer,
            "attention_mask": [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
        }
        
        return sample

    if not dataset_config or dataset_config["num_examples"] is None:
        dataset = dataset.map(
            lambda x: tokenize_add_label(x, dataset), 
            remove_columns=list(dataset.features)
        )
    else:
        dataset = dataset.map(
            lambda x: tokenize_add_label(x, dataset, num_examples=dataset_config["num_examples"]), 
            remove_columns=list(dataset.features)
        )

    return dataset

class CFADataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right" # during training, one always uses padding on the right

    def __call__(self, samples):
        dialogs,images = [],[]
        for sample in samples:
            image_list,sample_list = sample["texts"]
            if len(image_list) > 1:
                raise ValueError("Only support one image per sample")
            image = image_list[0].convert("RGB") # only use the first image
            dialog = []
            for sample_dict in sample_list:
                if not dialog:
                    # only append image to the first sentence
                    dialog += [
                    {"role":"user","content":[{"type": "image"},{"type": "text", "text": sample_dict["user"].strip()}]},
                    {"role":"assistant","content":[{"type": "text", "text": sample_dict["assistant"].strip()}]}
                ]
                
                else:
                    dialog += [
                    {"role":"user","content":[{"type": "text", "text": sample_dict["user"].strip()}]},
                    {"role":"assistant","content":[{"type": "text", "text": sample_dict["assistant"].strip()}]}
                ]
            dialogs.append(dialog)
            images.append([image])
        return tokenize_dialogs(dialogs,images, self.tokenizer)

def get_data_collator(tokenizer):
    return CFADataCollator(tokenizer)
