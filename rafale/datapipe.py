import os
import json
import hashlib
import warnings
from abc import ABC, abstractmethod

from datasets import load_dataset, DatasetDict

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from rafale.caches import DATA_CACHE_DIR

def compute_config_hash(serialized_config):
    return hashlib.sha256(serialized_config.encode('utf-8')).hexdigest()

class DataPipeline(ABC):
    """
    Base class

    A data pipeline is initiated with a path and some configurations parameters:

    Args:
    - dataset_path : local path to the dataset
    - collator : function to be applied when sending a batch through the dataloader


    Datasets should be saved using the following format:
    <dataset_name>/<split>.json

    Returns:
    """

    def __init__(self, **kwargs):
        self.name: str = kwargs["name"]
        self.tokenizer_name: str = kwargs["tokenizer_name"]
        self.is_prepared: bool = kwargs["is_prepared"]
        self.dataset_path: str = os.path.expanduser(kwargs["dataset_path"])

        self.shuffle_dataset: bool = kwargs["shuffle_dataset"]

        self.max_sequence_length: int = kwargs["max_sequence_length"]
        self.train_batch_size: int = kwargs["train_batch_size"]
        self.eval_batch_size: int = kwargs["eval_batch_size"]
        self.pad_inputs: bool = kwargs["pad_inputs"]
        self.pad_token_id: int = kwargs["pad_token_id"]

        self.input_id_key: str = kwargs["input_id_key"]
        self.shuffle_train: bool = kwargs["shuffle_train"]

        self.subset_key_mappings = kwargs["subset_key_mappings"]

        self.num_processes: int = kwargs["num_processes"]
        self.tokenizer = Tokenizer.from_pretrained(kwargs["tokenizer_path"])

        self.data_collator = None

        self.use_cached = False

        self.serialized_config = json.dumps(kwargs)
        self.config_dict = kwargs
        self.config_hash = compute_config_hash(self.serialized_config)

    def _dump_config(self, config_dict, output_dir):
        out_file = os.path.join(output_dir, "data_config.json")
        with open(out_file, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def _load(self):
        # either load directly from disk or from an hf dataset repo
        try:
            self.dataset = DatasetDict.load_from_disk(self.dataset_path)
        except:
            try:
                self.dataset = load_dataset(self.dataset_path)
                pass
            except:
                raise OSError(
                    f"Wrong dataset file and/or path configuration! path: {self.dataset_path}"
                )

    @abstractmethod
    def _prepare(self):
        """Perform all data preprocessing here: tokenization, chunking, truncation, etc. (EXCEPT padding!). Padding will be performed by
        the datacollator.
        """
        pass

    def _check_path(self):
        """make sure that the dataset has not already been parsed at location"""
        output_path = f"{self.name}_{self.tokenizer_name}_bs{self.train_batch_size}_len{self.max_sequence_length}_{self.config_hash}"

        assert DATA_CACHE_DIR[-1] == "/"
        save_path = os.path.abspath(os.path.join(DATA_CACHE_DIR, output_path))

        if os.path.isdir(DATA_CACHE_DIR):
            pass
        else:
            os.makedirs(DATA_CACHE_DIR)

        if os.path.isdir(save_path):
            warnings.warn(
                f"Dataset already exists at location:\n\t {save_path} \n ABORTING PREPARATION, USING CACHED DATASET!"
            )

            self.is_prepared = True
            self.use_cached = True

        return save_path

    def __call__(self):
        # returns a or multiple dataloaders
        self._load()

        self.dataloaders = {}

        if not self.is_prepared:
            cache_dataset_path = self._check_path()

        if type(self.dataset) == DatasetDict:
            for source_subset_key, dataloader_subset_key in self.subset_key_mappings.items():
                if dataloader_subset_key == "train":
                    shuffle = self.shuffle_train # shuffle training batches for multi-epoch training
                    batch_size = self.train_batch_size
                    if self.shuffle_dataset:
                        self.dataset[source_subset_key] = self.dataset[source_subset_key].shuffle(seed=42)
                else:
                    shuffle = False
                    batch_size = self.eval_batch_size

                # if the data is not ready to be passed to the dataloader
                if not self.is_prepared:
                    print(f"preparing subset {source_subset_key} -> {dataloader_subset_key}")
                    self.dataset[source_subset_key] = self._prepare(self.dataset[source_subset_key])

                if self.use_cached:
                    self.dataset = DatasetDict.load_from_disk(cache_dataset_path)

                self.dataloaders[dataloader_subset_key] = DataLoader(
                    self.dataset[source_subset_key],
                    collate_fn=self.data_collator,
                    batch_size=batch_size,
                )
                print(f"✅ Dataloader ready for subset {dataloader_subset_key}.")

            if not self.is_prepared:
                self.dataset.save_to_disk(cache_dataset_path)
                self._dump_config(self.config_dict, cache_dataset_path)
                print(f"✅ Saved prepared dataset at {cache_dataset_path}.")
        else:
            raise TypeError(
                f"dataset provided is type {type(self.dataset)}, but should be DatasetDict."
            )

        return self.dataloaders


class InferenceDatapipeline:
    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_path)

    def _tokenizer_templating(self, tokenizer, add_eos=True):
        if add_eos:
            tokenizer.post_processor = TemplateProcessing(
                single="<|endoftext|> $A",
                pair=None,
                special_tokens=[
                    ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
                ],
            )

            return tokenizer

    def __call__(self, input_str, use_template: bool = True):
        """
        tokenize input_str
        convert to torch tensor (batch_size=1)
        add the endoftext token
        """
        if use_template:
            self.tokenizer = self._tokenizer_templating(self.tokenizer)

        tokenized_inputs = {
            "input_ids": torch.LongTensor(
                self.tokenizer.encode(input_str).ids
            ).unsqueeze(dim=0)
        }

        return tokenized_inputs

    def ids_to_str(self, tensor):
        return self.tokenizer.decode(tensor.squeeze().detach().numpy())


class CausalCollator:
    def __init__(self, pad_token_id: int, input_id_key: str, pad_inputs: bool):
        self.pad_token_id = pad_token_id
        self.input_id_key = input_id_key
        self.pad_inputs = pad_inputs

    def __call__(self, features):
        # Extract the input IDs from the batch
        input_ids = [torch.tensor(example[self.input_id_key]) for example in features]

        # Pad the inputs if required
        if self.pad_inputs:
            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id
            )

        # Set the last token of each label sequence to pad_token_id to ignore loss for the last prediction
        # shift left the ids
        labels = [
            torch.cat([ids[1:], torch.tensor([self.pad_token_id])]) for ids in input_ids
        ]
        labels = torch.stack(labels, dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class MLMCollator:
    def __init__(
        self,
        mask_p: float = 0.15,
        whole_word_mask: bool = False,
        mask_span: bool = False,
        pad_token_id: int = -100,
        input_id_key: str = "input_ids",
        pad_inputs: bool = True,
    ):
        """masks some % of tokens for MLM objective"""
        raise NotImplementedError

class DefaultCollator:
    def __init__(
        self,
        pad_token_id: int = -100,
        input_id_key: str = "input_ids",
        label_key: str = "labels",
        pad_inputs: bool = True,
    ):
        """for task data where labels are already set"""
        self.pad_token_id = pad_token_id
        self.input_id_key = input_id_key
        self.label_key = label_key
        self.pad_inputs = pad_inputs

    def __call__(self, features):
        # Extract the input IDs from the batch
        input_ids = [torch.tensor(example[self.input_id_key]) for example in features]

        # Pad the inputs if required
        if self.pad_inputs:
            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id
            )

        # Set the last token of each label sequence to pad_token_id to ignore loss for the last prediction
        # shift left the ids
        labels = [torch.tensor(example[self.label_key]) for example in features]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
        }

class TinyStoriesCausalNeoX(DataPipeline):
    """This is a sample datapipeline for the TinyStories dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_collator = CausalCollator(
            pad_token_id=self.pad_token_id,
            input_id_key=self.input_id_key,
            pad_inputs=self.pad_inputs,
        )

    # TODO: figure out if they really only use endoftext for everything...
    def _tokenizer_templating(self, tokenizer, add_eos=True):
        if add_eos:
            tokenizer.post_processor = TemplateProcessing(
                single="<|endoftext|> $A",
                pair=None,
                special_tokens=[
                    ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
                ],
            )

            return tokenizer

    def _tokenize(self, example, tokenizer, key="text"):
        return {self.input_id_key: tokenizer.encode(example[key]).ids}

    def _group_and_chunk(self, examples, key="input_ids", block_size=None, pad=False):
        concatenated_tokens = sum(examples[key], [])
        total_length = len(concatenated_tokens)

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            "input_ids": [
                concatenated_tokens[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
        }

        return result

    def _prepare(self, data):
        """Preparation by first tokenizing the data, then grouping/chucking the corpus for efficient language modelling.
        """
        # apply functions above to dataset
        self.tokenizer = self._tokenizer_templating(self.tokenizer)

        data = data.map(
            lambda example: self._tokenize(example, self.tokenizer),
            remove_columns=data.column_names,
            num_proc=self.num_processes,
        )

        data = data.map(
            lambda example: self._group_and_chunk(
                example, block_size=self.max_sequence_length
            ),
            batched=True,
            num_proc=self.num_processes,
        )

        return data

class ImdbClsBERT(DataPipeline):
    """A pipeline for the imdb dataset for classification."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_key = "label"

        self.data_collator = DefaultCollator(
            input_id_key=self.input_id_key,
            label_key=self.label_key,
            pad_token_id=self.pad_token_id,
            pad_inputs=True,
        )

    def _tokenizer_templating(self, tokenizer, add_eos=True, max_length=None, truncate=False):
        if add_eos:
            tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair=None, # pair="[CLS] $A [SEP] $B:1 [SEP]:1", # not required for binary classification
                special_tokens=[
                    ("[CLS]", tokenizer.token_to_id("[CLS]")),
                    ("[SEP]", tokenizer.token_to_id("[SEP]")),
                ],
            )

            return tokenizer

    def _tokenize(
        self,
        example,
        tokenizer,
        key="text"
    ):

        return {
            self.input_id_key: tokenizer.encode(example[key]).ids,
            self.label_key: example[self.label_key],
        }

    def _prepare(self, data):
        """Tokenize by adding [CLS] and [SEP] tokens at the beginning and end of the text respectively.
        We also truncate to max length keeping only the first part of the sequence."""
        self.tokenizer = self._tokenizer_templating(self.tokenizer)

        # Truncation will include the special tokens, so set the true max_length of the model
        self.tokenizer.enable_truncation(self.max_sequence_length, strategy="only_first")

        data = data.map(
            lambda example: self._tokenize(example, self.tokenizer),
            remove_columns=data.column_names,
            num_proc=self.num_processes,
        )

        return data
