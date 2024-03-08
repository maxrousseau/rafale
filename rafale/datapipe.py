import os

from datasets import load_dataset, DatasetDict

from transformers import DataCollatorForLanguageModeling

from torch.utils.data import DataLoader


class DataPipeline:
    """
    Base class

    A data pipeline is initiated with a path and some configurations parameters:

    - path : local path to the dataset
    - collator_fn : function to be applied at parse

    Datasets should be saved using the following format:
    <dataset_name>/<split>.json
    """

    def __init__(self):
        self.path = None
        self.collator_fn = None
        self.padding = None
        self.truncation = None
        self.max_sequence_length = None
        self.shuffle_train = False
        self.batch_size = 4
        self.tokenizer = None

    def _load(self):
        try:
            self.dataset_pre = DatasetDict.load_from_disk(self.path)
        except:
            raise OSError(f"Wrong dataset file and/or path configuration!{self.path}")

    def _post_tokenize(self):
        None

    def __call__(self):
        # returns a or multiple dataloaders
        self._load()

        dataloaders = {}

        if type(self.dataset_pre) == DatasetDict:
            for subset in self.dataset_pre:
                if subset == "train":
                    shuffle = self.shuffle_train
                else:
                    shuffle = False

                # @DEBUG
                self.dataset_pre[subset] = self.dataset_pre[subset].select(range(10))
                _set = self.dataset_pre[subset].map(
                    lambda example: self._tokenize(
                        example,
                    )
                )
                _set = self._post_tokenize(_set)
                print(len(_set["input_ids"][0]))
                print(_set)
                dataloaders[subset] = DataLoader(
                    _set,
                    collate_fn=self.collator_fn,  # DEBUG
                    batch_size=self.batch_size,
                    shuffle=shuffle,  # DEBUG
                )
        else:
            # process a single set
            None

        return dataloaders


# TODO:
# download 10k wiki random subset (or "simple") - shuffle then download, do that in a colab notebook...
# then apply MLM loading to it for testing...


# datapipe is simple, you call the function and get the dataloader(s) you need for running
# a debug_mode flag can be included (single batch)
class WikiMLMPipe(DataPipeline):
    """a first testing pipeline for wikipedia for MLM"""

    def __init__(self, **kwargs):
        self.path = kwargs["path"]
        self.padding = kwargs["padding"]  # "max_length"
        self.truncation = kwargs["truncation"]  # "max_length"
        self.max_sequence_length = kwargs["max_sequence_length"]  # 512
        self.shuffle_train = kwargs["shuffle_train"]  # False
        self.batch_size = kwargs["batch_size"]
        self.tokenizer = kwargs["tokenizer"]
        self.collator_fn = DataCollatorForLanguageModeling(tokenizer=self.tokenizer)

    def _post_tokenize(self, dataset):
        return dataset.remove_columns(["url", "title", "text", "id"])

    def _tokenize(
        self,
        examples,
    ):
        source_tokenized = self.tokenizer(
            examples["text"],
            padding=self.padding,
            max_length=self.max_sequence_length,
            truncation=self.truncation,
        )

        batch = {k: v for k, v in source_tokenized.items()}

        return batch


"""
args = {"path" : "~/code/data/enwiki1m", "truncation": True, "max_sequence_length": 128, "shuffle_train" : False,
"batch_size":4, "padding": "max_length", "tokenizer" : tokenizer}

wikipipe = WikiMLMPipe(**args)
dloaderz = wikipipe()

next(iter(dloaderz["train"]))

"""

# class WikiSLaMPipe(DataPipeline):
