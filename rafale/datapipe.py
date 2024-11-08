import os
import warnings
from abc import ABC, abstractmethod

from datasets import load_dataset, DatasetDict

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from rafale.caches import DATA_CACHE_DIR


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

        self.max_sequence_length: int = kwargs["max_sequence_length"]
        self.train_batch_size: int = kwargs["train_batch_size"]
        self.eval_batch_size: int = kwargs["eval_batch_size"]
        self.pad_inputs: bool = kwargs["pad_inputs"]
        self.pad_token_id: int = kwargs["pad_token_id"]
        self.input_id_key: str = kwargs["input_id_key"]
        self.shuffle_train: bool = kwargs["shuffle_train"]

        self.num_processes: int = kwargs["num_processes"]
        self.tokenizer = Tokenizer.from_pretrained(kwargs["tokenizer_path"])

        self.data_collator = None

        self.use_cached = False

    def _load(self):
        try:
            self.dataset = DatasetDict.load_from_disk(self.dataset_path)
        except:
            pass
        try:
            self.dataset = load_dataset(self.dataset_path)
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
        output_path = f"{self.name}_{self.tokenizer_name}_bs{self.train_batch_size}_len{self.max_sequence_length}"

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
            for subset in self.dataset:
                if subset == "train":
                    shuffle = self.shuffle_train
                    batch_size = self.train_batch_size
                else:
                    shuffle = False
                    batch_size = self.eval_batch_size

                # if the data is not ready to be passed to the dataloader
                if not self.is_prepared:
                    print(f"preparing subset {subset}")
                    self.dataset[subset] = self._prepare(self.dataset[subset])

                if self.use_cached:
                    self.dataset = DatasetDict.load_from_disk(cache_dataset_path)

                self.dataloaders[subset] = DataLoader(
                    self.dataset[subset],
                    collate_fn=self.data_collator,
                    batch_size=batch_size,
                )
                print(f"✅ Dataloader ready for subset {subset}.")

            if not self.is_prepared:
                self.dataset.save_to_disk(cache_dataset_path)
                print(f"✅ Saved prepared dataset at {cache_dataset_path}.")
        else:
            raise TypeError(
                f"self.dataset is type {type(self.dataset)}, but should be DatasetDict."
            )

        return self.dataloaders


class InferenceDatapipeline:
    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_path)

    def _tokenizer_templating(self, tokenizer, add_eos=True):
        if add_eos:
            tokenizer.post_processor = TemplateProcessing(
                single="<|endoftext|> $A",
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
        return ifdp.tokenizer.decode(tensor.squeeze().detach().numpy())


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
        pass


class DefaultCollator:
    def __init__(
        self,
        pad_token_id: int = -100,
        input_id_key: str = "input_ids",
        pad_inputs: bool = True,
    ):
        """for task data where labels are already set"""
        pass


class TinyStoriesCausalNeoX(DataPipeline):
    """This is sample datapipelin for the TinyStories dataset.


    This dataset is prepared for causal language modelling using the gpt neox tokenizer (eleutherai). We

    Usage:
        ts_dict = {
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
        ts_dpipe = TinyStoriesCausalNeoX(**ts_dict)
        dataloaders = ts_causal()

    Args:

    Returns:

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
                single="$A <|endoftext|>",
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
        """"""

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


class TinyStoriesMLM(DataPipeline):
    """ """

    pass


class ImdbCLS(DataPipeline):
    pass


'''
class ImdbClsPipe(DataPipeline):
    """A pipeline for the imdb dataset for """

    def __init__(self, **kwargs):
        self.path = kwargs["path"]
        self.name = kwargs["name"] # name
        self.is_tokenized = kwargs["is_tokenized"]

        self.padding = kwargs["padding"]  # "max_length"
        self.max_sequence_length = kwargs["max_sequence_length"]  # 512

        self.shuffle_train = kwargs["shuffle_train"] # False
        self.batch_size = kwargs["batch_size"]
        self.tokenizer = kwargs["tokenizer"]
        self.collator_fn = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_sequence_length,
            return_tensors='pt'
        )

        self.data = datasets.DatasetDict.load_from_disk(self.path)

    def _post_tokenize(self, dataset):
        return dataset.remove_columns(["text"])

    def _tokenize(
        self,
        examples,
    ):
        source_tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_sequence_length,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        batch = {k: v for k, v in source_tokenized.items()}

        return batch

    def _map_tokenize(self, subsets=None):
        # tokenize
        print("tokenizing training")
        self.data["train"] = self.data["train"].map(self._tokenize, batched=True)
        self.data["train"] = self.data["train"].remove_columns("text")

        print("tokenizing test")
        self.data["test"] = self.data["test"].map(self._tokenize, batched=True)
        self.data["test"] = self.data["test"].remove_columns("text")


    def _save_tokenized(self):
        # preprocess
        self.path += "_tokenized"
        print(f"saving tokenized data to disk at location : {self.path}")
        assert os.path.isdir(self.path) == False
        self.data.save_to_disk(self.path)

    def string_to_tokens(self, input_str):
        # tokenize
        tensor = self._tokenize({"text": input_str})

        return self.collator_fn(tensor)

    def __call__(self, subsets = ["train", "test"]):
        dataloaders = {}

        if self.is_tokenized:
            print("data tokenized")
        else:
            self._map_tokenize()
            self._save_tokenized()

        for _set in subsets:
            if _set == "train":
                shuffle = self.shuffle_train
            else:
                shuffle = False

            dataloaders[_set] = DataLoader(
                        self.data[_set],
                        collate_fn=self.collator_fn,  # DEBUG
                        batch_size=self.batch_size,
                        shuffle=shuffle,  # DEBUG
                    )
        return dataloaders
'''
