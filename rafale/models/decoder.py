#!/usr/bin/env python
from typing import Optional
import torch

from torch import nn
from torch import Tensor

# from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.functional import scaled_dot_product_attention

###############################################################################
#                 simple implementation of GPT building blocks                #
###############################################################################


class RoPE(nn.Module):
    @classmethod
    def precompute_freqs_cis(
        cls,
        seq_len: int,
        n_elem: int,
        base: int = 10000,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tensor:
        """
        Computes the cos and sin angles to be applied to the token vectors.

        We begin by computing thetas (freqs) across each dimension pair (P=D/2) for the whole sequence length (L).
        Then we convert this matrix of shape LP into complex numbers of the same shape.
        Finally the real and imaginary parts of these complex numbers are stored in a stacked matrix and returned.

        Args:
            seq_len (int): sequence length of the input (use the maximum sequence length)
            n_elem (int): hidden dimension of the model (D)
            base (int): default 10000

        Returns:
            Tensor # of shape LP2
        """
        freqs = 1.0 / (
            base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
        )  # shape is D/2 or n_elem/2
        t = torch.arange(seq_len, device=freqs.device)  # shape L
        freqs = torch.outer(t, freqs)  # outer product yields matrix of shape LD/2
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # shape LD/2
        # torch.ones_like simply returns a tensor of ones with the same shape
        # torch.polar creates a complex tensor whose elements are Cartesian coordinates corresponding to the polar
        # coordinates with absolute value abs (ones_like) and angle angle (freqs).
        cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
        # stack the real and imaginary part of the complex tensor shape is L-D/2-2
        return cache.to(dtype=dtype)

    @classmethod
    def apply_rotary_emb(cls, x: Tensor, freqs_cis: Tensor) -> Tensor:
        """
        Applies Rotary Position Embedding (RoPE) to the input tensor.

        shapes:
        B: batch size
        N: number of heads
        L: sequence length
        d: head dimension
        O: flat dimension (1)
        P: paired dimension (d/2)


        This function reshapes the input tensor `x` and the frequency tensor `freqs_cis`
        to prepare them for the application of RoPE. For each pair of hidden dimensions
        (m, n) in `x`, the following transformations are applied:

            m' = m * real_part(freqs_cis) - n * imaginary_part(freqs_cis)
            n' = n * real_part(freqs_cis) + m * imaginary_part(freqs_cis)

        After applying the transformations, the tensor is reshaped back to its original shape.

        Args:
            x (Tensor): The input tensor of shape (..., 2 * hidden_dim), where the last dimension
                        consists of pairs of (m, n) values.
            freqs_cis (Tensor): The complex-valued frequency tensor of shape (..., 1, 2), where the
                                last dimension contains (real_part, imaginary_part) values corresponding
                                to each frequency.

        Returns:
            Tensor: The tensor after applying RoPE, with the same shape as the input tensor `x`.
        """

        x = x.transpose(1, 2)  # flip back the seq_len and num_heads
        x_BLNP2 = x.float().reshape(*x.shape[:-1], -1, 2)

        freqs_cis_1L1P2 = freqs_cis.view(1, x_BLNP2.size(1), 1, x_BLNP2.size(3), 2)

        x_out2 = torch.stack(
            [
                x_BLNP2[..., 0] * freqs_cis_1L1P2[..., 0]
                - x_BLNP2[..., 1] * freqs_cis_1L1P2[..., 1],
                x_BLNP2[..., 1] * freqs_cis_1L1P2[..., 0]
                + x_BLNP2[..., 0] * freqs_cis_1L1P2[..., 1],
            ],
            -1,
        )

        x_out2 = x_out2.flatten(3)
        x_BNLd = x_out2.transpose(1, 2)

        return x_BNLd.type_as(x)


class DecoderEmbedding(nn.Module):
    """simply an input projection of the tokens here, since rotary position encodings are used, makes things simpler"""

    def __init__(self, config):
        super().__init__()
        # nn.Embedding is just a lookup table,
        self.input_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            # padding_idx=pad_token_id, # not specified for causal GPTNeoX... ? @TODO :: i think padding is handled by
            # the attention mask...
        )
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x):
        x = self.input_embeddings(x)
        return self.dropout(x)


#        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) #* @NOTE :: interesting no layer norm in the
#        pythia reference HF implementation, is this a thing for all causal decoders? Ans: since they use input
#        layernorm and post attn layer norm this isn't necessary here


class DecoderAttentionRotary(nn.Module):
    """
    Attention with rotary position embedding

    Tensor dimension names:
    - B batch size
    - L sequence length
    - H number of attention heads
    - D embedding dimension
    - d attention head dimension D//H
    - F feedforward dimension

    """

    def __init__(self, config):
        super().__init__()

        self.head_dim = config.embed_dim // config.num_heads
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim

        # set bias to True or False (@TODO)
        self.query_key_value = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.dropout_p = config.attention_dropout

    def _split_heads(self, tensor: Tensor):
        """
        Splits hidden dim into attn_head_size and num_attention_heads

        # input tensor: [bs, seq_len, hidden_size]
        # returns: [bs, num_attention_heads, seq_len, attn_head_size]
        """
        batch_size = tensor.size(0)

        return tensor.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def _merge_heads(self, tensor: Tensor):
        """
        input tensor: [bs. num_attention_heads, seq_len, attn_head_size]
        returns: [bs, seq_len, hidden_size]
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(
            tensor.size(0),
            tensor.size(1),
            self.num_heads * self.head_dim,
        )
        # -> [bs, seq_len, hidden_size]
        return tensor

    def forward(self, x_BLD, freqs_cis, mask):
        seq_len = x_BLD.size(1)
        assert freqs_cis is not None
        # Slice the precomputed freqs_cis based on actual seq_len
        print(seq_len)
        print(freqs_cis)
        sliced_freqs_cis = freqs_cis[:seq_len, :, :]

        # projection
        q_BLD, k_BLD, v_BLD = self.query_key_value(x_BLD).split(self.embed_dim, dim=-1)

        q_BHLd = self._split_heads(q_BLD)
        k_BHLd = self._split_heads(k_BLD)
        v_BHLd = self._split_heads(v_BLD)

        # apply rotary embeddings
        q_BHLd = RoPE.apply_rotary_emb(q_BHLd, sliced_freqs_cis)
        k_BHLd = RoPE.apply_rotary_emb(k_BHLd, sliced_freqs_cis)

        # compute attention
        # attn_out_BLHd = flex_attention(query, key, value, block_mask=mask)
        attn_out_BLHd = scaled_dot_product_attention(
            q_BHLd, k_BHLd, v_BHLd, is_causal=True, dropout_p=self.dropout_p
        )

        attn_out_BLD = self._merge_heads(attn_out_BLHd)

        return attn_out_BLD


class DecoderFeedForward(nn.Module):
    """
    Tensor dimension names:
    - B batch size
    - L sequence length
    - H number of attention heads
    - D embedding dimension
    - d attention head dimension D//H
    - F feedforward dimension

    """

    def __init__(self, config):
        super().__init__()
        self.ff_in = nn.Linear(config.embed_dim, config.ff_dim)
        self.gelu = nn.GELU()
        self.ff_out = nn.Linear(config.ff_dim, config.embed_dim)

    def forward(self, x_BLD):
        x_BLF = self.ff_in(x_BLD)
        x_BLF = self.gelu(x_BLF)
        out_BLD = self.ff_out(x_BLF)
        return out_BLD


class DecoderBlock(nn.Module):
    """ """

    def __init__(self, config):
        super().__init__()
        self.attention = DecoderAttentionRotary(config)
        self.feed_forward = DecoderFeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)
        self.attention_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)

    def forward(self, x, freqs_cis, mask):
        # @NOTE :: this i different from BERT*** norm then add vs add then norm
        assert freqs_cis is not None
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class DecoderWrapper(nn.Module):
    """
    self explanatory
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embeddings = DecoderEmbedding(config)

        self.layers = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.num_blocks)
        )
        self.final_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.causal_mask: Optional[Tensor] = None

        self.max_batch_size = -1
        self.max_seq_length = config.max_pos_embedding

    def setup_caches(self):
        head_dim = self.config.embed_dim // self.config.num_heads
        dtype = self.output.weight.dtype

        self.freqs_cis = RoPE.precompute_freqs_cis(
            seq_len=self.config.max_pos_embedding,
            n_elem=self.config.embed_dim // self.config.num_heads,
            base=self.config.rotary_emb_base,
            dtype=dtype,
        )

        # @TODO :: figure out why the nightly build thing...
        # self.causal_mask = create_block_mask(
        #     causal,
        #     B=None,
        #     H=None,
        #     Q_LEN=self.max_seq_length,
        #     KV_LEN=self.max_seq_length,
        # )  # batch and heads will be broadcast

    def forward(self, idx: Tensor):
        # if self.freqs_cis is None or self.causal_mask is None:
        if self.freqs_cis is None:
            self.setup_caches()  # Caches must be initialized first

        # mask = self.causal_mask[None, None, input_pos]  # we crop the excessive length?
        mask = self.causal_mask  # @TODO :: fix this when we add flex attention!
        freqs_cis = self.freqs_cis

        x = self.token_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, mask)
        x = self.final_norm(x)
        logits = self.output(x)
        return logits
