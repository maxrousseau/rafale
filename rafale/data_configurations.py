#!/usr/bin/env python
# data configurations for your experiments ###################################

MINI_TINYSTORIES = {
    "name": "tinystories_testing",
    "tokenizer_name": "neox",
    "is_prepared": False,
    "input_id_key": "input_ids",
    "batch_size": 16,
    "shuffle_train": False,
    "dataset_path": "~/code/data/micro_tinystories",
    "tokenizer_path": "EleutherAI/pythia-14m",
    "max_sequence_length": 128,
    "pad_token_id": -100,
    "pad_inputs": True,
    "is_prepared": False,
}

TINYSTORIES = {
    "name": "tinystories",
    "num_processes": 8,
    "tokenizer_name": "neox",
    "is_prepared": False,
    "input_id_key": "input_ids",
    "train_batch_size": 1024,
    "eval_batch_size": 16,
    "shuffle_train": False,
    "dataset_path": "~/code/data/TinyStories",
    "tokenizer_path": "EleutherAI/pythia-14m",
    "max_sequence_length": 512,
    "pad_token_id": -100,
    "pad_inputs": True,
    "is_prepared": False,
}

MINIPILE = {}

FINEWEB_EDU = {}
