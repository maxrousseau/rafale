import torch

from encoder import EncoderWrapper
from dataclasses import dataclass

# from composer.models import ComposerModel


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
    def __init__(self, config):
        super().__init__(config)
        self.embedding_layer.forward = self.roberta_embedding_forward
        # monkey patched forward method to fix the position_id embedding without changing the original encoder embedding class.

    def mlm_hook(self):
        """TBD"""
        None

    def roberta_embedding_forward(self, input_ids, token_type_ids):
        position_ids = self.create_position_ids_from_input_ids(input_ids, 1)

        # we assume absolute positional encoding here like in the original BERT and sum everything up
        W = self.embedding_layer.word_embeddings(input_ids)
        P = self.embedding_layer.position_embeddings(position_ids)
        T = self.embedding_layer.token_type_embeddings(token_type_ids)

        E = W + P + T
        E = self.embedding_layer.LayerNorm(E)
        E = self.embedding_layer.dropout(E)

        return E

    def create_position_ids_from_input_ids(
        self, input_ids, padding_idx, past_key_values_length=0
    ):
        """
        MAX NOTE: from the huggingface implementation, they use a different method to create the positon_ids in roberta than
        bert. whithout this the model breaks... simply modifies the method used to cast the array.

        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (
            torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
        ) * mask
        return incremental_indices.long() + padding_idx
