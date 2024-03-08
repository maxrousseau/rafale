from tokenizers import Tokenizer
from encoder import EncoderTransformer


class RobertaWrapper(EncoderTransformer):
    def __init__(
        self,
    ):
        self.roberta_tokenizer = Tokenizer.from_file("roberta/tokenizer.json")
