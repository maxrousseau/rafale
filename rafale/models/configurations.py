from dataclasses import dataclass

"""
to simplify model loading add a configuration for the pre-trained weight loading using safetensors instead of loading
the full model.
> then save to a folder named ".pretrained/" in this directory
"""


@dataclass
class BertConfig:
    embed_dim: int = 768
    vocab_size: int = 30522
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    num_heads: int = 12
    ff_dim: int = 3072
    max_pos_embedding: int = 512
    layer_norm_eps: float = 1e-12
    num_blocks: int = 12
    pad_token_id: int = 0
    num_token_type: int = 2
    fast_attention: bool = (
        False  # use xformers (todo: add FlashAttention2), NOT IMPLEMENTED*
    )


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
    fast_attention: bool = False


@dataclass
class Pythia14MConfig:
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 512
    hidden_act: str = "gelu"
    max_pos_embedding: int = 2048
    vocab_size: int = 50304

    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1

    layer_norm_eps: float = 1e-05
    num_blocks: int = 6
    # pad_token_id: int = 1

    bos_token_id: int = 0
    eos_token_id: int = 0
    fast_attention: bool = False  # use xformers (todo: add FlashAttention2)

    rotary_emb_base: int = 10000  # @TODO read a breakdown of rotatry pos embeddings and figure out what this does
    rotary_pct: float = 0.25  # what is this?...

    tie_word_embeddings: bool = False


# @TODO :: delete below when we tested the config
# {
#   "architectures": [
#     "GPTNeoXForCausalLM"
#   ],
#   "bos_token_id": 0,
#   "classifier_dropout": 0.1,
#   "eos_token_id": 0,
#   "hidden_act": "gelu",
#   "hidden_size": 128,
#   "initializer_range": 0.02,
#   "intermediate_size": 512,
#   "layer_norm_eps": 1e-05,
#   "max_position_embeddings": 2048,
#   "model_type": "gpt_neox",
#   "num_attention_heads": 4,
#   "num_hidden_layers": 6,
#   "rotary_emb_base": 10000,
#   "rotary_pct": 0.25,
#   "tie_word_embeddings": false,
#   "torch_dtype": "float16",
#   "transformers_version": "4.29.2",
#   "use_cache": true,
#   "use_parallel_residual": true,
#   "vocab_size": 50304
# }
#
#
