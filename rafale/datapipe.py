import os

from datasets import load_dataset, DatasetDict

from tokenizers import Tokenizer

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


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


def tokenize(example, tokenizer, key="text"):
    return {"input_ids": tokenizer.encode(example[key]).ids}


# figure out if they really only use endoftext for everything...
def get_pythia_tokenizer(repo="EleutherAI/pythia-14m", add_eos=True):
    tokenizer = Tokenizer.from_pretrained(repo)

    if add_eos:
        tokenizer.post_processor = TemplateProcessing(
            single="$A <|endoftexqt|>",
            special_tokens=[
                ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
            ],
        )

    return tokenizer


def group_and_chunk(examples, key="input_ids", block_size=128, pad=False):
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


# ts_1k = ts_1k.map(
#     lambda example: tokenize(example, neox_tokenizer), remove_columns=ts_1k.column_names
# )

# lm_ = ts_mini_tokenized.map(lambda example: group_and_chunk(example, block_size=128), batched=True)
# ccollator = CausalCollator(pad_token_id=-100, input_id_key="input_ids", pad_inputs=False)
# mini_dl = DataLoader(lm_, collate_fn=ccollator, batch_size=4, shuffle=False)

# 1) [x] tokenize
# 2) [x] concat and split w/ block size (pad w/ collator)
# 3) [ ] save to disk {source}_{tokname}_bs{int}_len{int}
# 3) [x] data_collator: *next* pad (if desired), label shift right and return torch tensor # HF: does this in the model...
# 4) [ ]


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

        # Convert input_ids to tensor if not padded yet
        if not self.pad_inputs:
            input_ids = torch.stack(input_ids, dim=0)

        # Stack the labels as well if padding was applied
        if self.pad_inputs:
            labels = torch.stack(labels, dim=0)

        # Return the dictionary in the format used for training
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class TinyStories(DataPipeline):
    """ """

    def __init__(self, args):
        super().__init__()


ts_dict = {
    path: "~/code/data/tinystories",
    tokenizer_path: "EleutherAI/pythia-14m",
    padding: "max_length",
    truncation: "max_length",
}


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
            return_token_type_ids=True,
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
