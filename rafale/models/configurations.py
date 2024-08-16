"""
to simplify model loading add a configuration for the pre-trained weight loading using safetensors instead of loading
the full model.
> then save to a folder named ".pretrained/" in this directory
"""


@dataclass
class BertConfig:
    embed_dim: int = 768
    vocab_size: int = 30522  # could usage would be to 30522 + num_extra_tokens
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    num_heads: int = 12
    ff_dim: int = 3072
    max_pos_embedding: int = 512
    layer_norm_eps: float = 1e-12
    num_blocks: int = 12
    pad_token_id: int = 0
    num_token_type: int = 2
    fast_attention: bool = False  # use xformers (todo: add FlashAttention2)


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
