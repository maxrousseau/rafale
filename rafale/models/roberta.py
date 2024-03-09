from transformers import RobertaTokenizer

# from encoder import EncoderWrapper  # debug
from dataclasses import dataclass


@dataclass
class RobertaConfig:
    embed_dim: int = 768
    vocab_size: int = 50265
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    num_heads: int = 12
    ff_dim: int = 3072
    max_pos_embedding: int = 514
    layer_norm_eps: float = 1e-05
    num_blocks: int = 12
    pad_token_id: int = 1
    num_token_type: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    fast_attention: bool = False  # use xformers (todo: add FlashAttention2)


class RobertaMLM(EncoderWrapper):
    def __init__(
        self,
        weights=None,
        config=None,
    ):
        super().__init__(config)
        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

        # load the weights (need to be converted first)

        # compute a fwd pass
