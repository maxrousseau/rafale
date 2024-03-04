import os
from datasets import load_dataset


class DataPipeline:
    """
    Base class

    A data pipeline is initiated with a path and some configurations parameters:

    - path : local path to the dataset
    - collator_fn : function to be applied at parse

    Datasets should be saved using the following format:
    <dataset_name>/<split>.json
    """

    def __init__(self, path):
        self.path = path
        self.collator_fn = None

    def _load(self):
        try:
            self.dataset_pre = load_dataset(os.path.abspath(self.path))
        except:
            raise OSError("Wrong dataset file and/or path configuration!")

    def _tokenize(
        example, tokenizer=None, padding=None, truncation=None, max_sequence_length=None
    ):
        """ """


# TODO:
# download 10k wiki random subset (or "simple") - shuffle then download, do that in a colab notebook...
# then apply MLM loading to it for testing...


# datapipe is simple, you call the function and get the dataloader(s) you need for running
# a debug_mode flag can be included (single batch)
class WikiMLMPipe(DataPipeline):
    """a first testing pipeline for wikipedia for MLM"""

    def __init__(self):
        super()

    def __call__(self, tokenizer):
        # returns a or multiple dataloaders
        self._load()
        for subset in self.dataset_pre:
            subset.map(lambda example: self._tokenize(example))

        return None


# class WikiSLaMPipe(DataPipeline):
