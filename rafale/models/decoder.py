#!/usr/bin/env python
from typing import Optional
import torch

from torch import nn
from torch import Tensor

import torch.nn.functional as F

from torch.nn.functional import scaled_dot_product_attention

from composer.models import ComposerModel

###############################################################################
#                 simple implementation of GPT building blocks                #
###############################################################################


class NeoXRoPE(nn.Module):
    @classmethod
    def precompute_sin_cos_cache(cls, dim=None, seq_len=None, base=10000):
        """Computes the cos and sin angles to be applied to the token vectors.

        We begin by computing thetas (freqs) across each dimension pair (P=D/2) for the whole sequence length (L).
        Then we convert this matrix of shape LP into complex numbers of the same shape.
        Finally the real and imaginary parts of these complex numbers are stored in a stacked matrix and returned.

        Args:
            dim (int): number of features dimension per token to apply rotations to (d*rotary_pct)
            seq_len (int): sequence length of the input (use the maximum sequence length)
            base (int): default 10000

        Returns:
            Tensor # of shape [1,1,L,R]

        Tensor dimension names:
        - B batch size
        - L sequence length
        - H number of attention heads
        - D embedding dimension
        - d attention head dimension D//H
        - F feedforward dimension
        - R rotary dimensions (d*rotary_pct)
        """

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        t = torch.arange(seq_len, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        return cos_cached, sin_cached

    @classmethod
    def apply_rotary_pos_emb(cls, q_BHLR, k_BHLR, cos, sin):
        """Applies the rotation to the input queries and key features."""

        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=x1.ndim - 1)

        return (q_BHLR * cos) + (rotate_half(q_BHLR) * sin), (k_BHLR * cos) + (
            rotate_half(k_BHLR) * sin
        )


class DecoderEmbedding(nn.Module):
    """Simply an input projection of the tokens here, since rotary position encodings are used.

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

        self.input_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x_BL):
        x_BLD = self.input_embeddings(x_BL)
        return self.dropout(x_BLD)


class DecoderAttentionRotary(nn.Module):
    """
    Attention with rotary position embedding.

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
        self.rotary_ndims = int(self.head_dim * config.rotary_pct)

        self.attention_bias = True  # @TODO: set bias to True or False from config.
        self.query_key_value = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.dropout_p = config.attention_dropout
        self.norm_factor = self.head_dim**-0.5

    def _split_heads(self, tensor: Tensor):
        """
        Splits hidden dim into attn_head_size and num_attention_heads

        # input tensor: [bs, seq_len, hidden_size]
        # returns: [bs, num_attention_heads, seq_len, attn_head_size]
        """
        batch_size = tensor.size(0)

        return (
            tensor.view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
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
        ).contiguous()
        # -> [bs, seq_len, hidden_size]
        return tensor

    def forward(self, x_BLD, freqs_cis, mask):
        if not self.training:
            self.dropout_p = 0
        bsz, seq_len, _ = x_BLD.size()

        assert freqs_cis is not None

        # projections
        qkv = self.query_key_value(x_BLD)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_heads, 3 * self.head_dim)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        q_BHLd = qkv[..., : self.head_dim].permute(0, 2, 1, 3)
        k_BHLd = qkv[..., self.head_dim : 2 * self.head_dim].permute(0, 2, 1, 3)
        v_BHLd = qkv[..., 2 * self.head_dim :].permute(0, 2, 1, 3)

        # Slice the precomputed freqs_cis based on actual seq_len --> [1, 1, seq_len, R]
        cos = freqs_cis[0][:, :, :seq_len, :]
        sin = freqs_cis[1][:, :, :seq_len, :]

        q_rot = q_BHLd[..., : self.rotary_ndims]
        q_pass = q_BHLd[..., self.rotary_ndims :]
        k_rot = k_BHLd[..., : self.rotary_ndims]
        k_pass = k_BHLd[..., self.rotary_ndims :]

        q_rot, k_rot = NeoXRoPE.apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        q_BHLd = torch.cat((q_rot, q_pass), dim=-1)
        k_BHLd = torch.cat((k_rot, k_pass), dim=-1)

        # compute attention
        attn_out_BHLd = scaled_dot_product_attention(
            q_BHLd,
            k_BHLd,
            v_BHLd,
            is_causal=True,
            scale=self.norm_factor,
            dropout_p=self.dropout_p,
        )

        attn_out_BLD = self._merge_heads(attn_out_BHLd)

        attn_out_BLD = self.dense(attn_out_BLD)

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
    """A single trasnformer decoder block/layer.

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
        self.attention = DecoderAttentionRotary(config)
        self.feed_forward = DecoderFeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)
        self.attention_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)

    def forward(self, x_BLD, freqs_cis, mask, parallel_residual=True, use_cache=True):
        assert freqs_cis is not None

        if parallel_residual:
            out_BLD = (
                x_BLD
                + self.attention(self.attention_norm(x_BLD), freqs_cis, mask)
                + self.feed_forward(self.ffn_norm(x_BLD))
            )
        else:
            h_BLD = x_BLD + self.attention(self.attention_norm(x_BLD), freqs_cis, mask)
            out_BLD = h_BLD + self.feed_forward(self.ffn_norm(h_BLD))

        return out_BLD


class DecoderWrapper(ComposerModel):
    """Full model wrapper for causal language modelling."""

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

        self.rotary_pct = 0.25

        self.vocab_size = config.vocab_size

        self.ce_loss = nn.CrossEntropyLoss()

    def setup_caches(self):
        head_dim = self.config.embed_dim // self.config.num_heads
        dtype = self.output.weight.dtype

        head_size = self.config.embed_dim // self.config.num_heads
        rotary_ndims = int(head_size * self.rotary_pct)
        self.cos, self.sin = NeoXRoPE.precompute_sin_cos_cache(
            dim=rotary_ndims,
            seq_len=self.config.max_pos_embedding,
        )

        self.freqs_cis = (self.cos, self.sin)

        self.causal_mask = torch.ones(
            self.config.max_pos_embedding,
            self.config.max_pos_embedding,
            dtype=torch.bool,
        )
        self.causal_mask = torch.tril(self.causal_mask)
        self.causal_mask = self.causal_mask.unsqueeze(0).unsqueeze(0)

    def forward(self, batch: Tensor):
        # if self.freqs_cis is None or self.causal_mask is None:
        if self.freqs_cis is None:
            self.setup_caches()  # Caches must be initialized first

        mask = self.causal_mask  # not actually used for now...
        freqs_cis = self.freqs_cis

        idx = batch["input_ids"]

        x = self.token_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, mask)
        x = self.final_norm(x)
        logits = self.output(x)

        return logits

    def loss(self, outputs, batch):
        targets = batch["labels"]

        return self.ce_loss(outputs.view(-1, self.vocab_size), targets.view(-1))
