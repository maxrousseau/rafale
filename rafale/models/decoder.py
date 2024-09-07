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


# @BUG problem is here definetely... STILL BROKEN
class NeoXRoPE(nn.Module):
    """BROKEN!"""

    @classmethod
    def precompute_sin_cos_cache(cls, dim=None, seq_len=None, base=10000):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        t = torch.arange(seq_len, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        return cos_cached, sin_cached

    # caches are identical so it is probably the apply method which is causing the problem!

    # @torch.jit.script
    @classmethod
    def apply_rotary_pos_emb(cls, q, k, cos, sin):
        """q shape BHLd"""

        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=x1.ndim - 1)

        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class LlamaRoPE(nn.Module):
    @classmethod
    def precompute_freqs_cis(
        cls,
        seq_len: int,
        n_elem: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
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

        x = x.transpose(1, 2).contiguous()  # flip back the seq_len and num_heads
        x_BLNP = x.float().reshape(*x.shape[:-1], -1, 2)

        freqs_cis_1L1P2 = freqs_cis.view(1, x_BLNP.size(1), 1, x_BLNP.size(3), 2)

        x_out2 = torch.stack(
            [
                x_BLNP[..., 0] * freqs_cis_1L1P2[..., 0]
                - x_BLNP[..., 1] * freqs_cis_1L1P2[..., 1],
                x_BLNP[..., 1] * freqs_cis_1L1P2[..., 0]
                + x_BLNP[..., 0] * freqs_cis_1L1P2[..., 1],
            ],
            -1,
        )

        x_out2 = x_out2.flatten(3)
        x_BNLd = x_out2.transpose(1, 2).contiguous()
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

        # ADD ROTARY PCT TO CONFIG !@TODO
        self.rotary_ndims = int(self.head_dim * 0.25)

        # set bias to True or False (@TODO)
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
        # Slice the precomputed freqs_cis based on actual seq_len @NOTE LLAMA
        # sliced_freqs_cis = freqs_cis[:seq_len, :, :]

        # projection
        # q_BLD, k_BLD, v_BLD = self.query_key_value(x_BLD).split(self.embed_dim, dim=-1)
        qkv = self.query_key_value(x_BLD)

        # fixing shapes ###############################################################
        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_heads, 3 * self.head_dim)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        q_BHLd = qkv[..., : self.head_dim].permute(0, 2, 1, 3)
        k_BHLd = qkv[..., self.head_dim : 2 * self.head_dim].permute(0, 2, 1, 3)
        v_BHLd = qkv[..., 2 * self.head_dim :].permute(0, 2, 1, 3)
        # .... ################################################################

        # below, [1, 1, seq_len, hdim] - expects shape BHLd
        cos = freqs_cis[0][:, :, :seq_len, :]
        sin = freqs_cis[1][:, :, :seq_len, :]

        q_rot = q_BHLd[..., : self.rotary_ndims]
        q_pass = q_BHLd[..., self.rotary_ndims :]
        k_rot = k_BHLd[..., : self.rotary_ndims]
        k_pass = k_BHLd[..., self.rotary_ndims :]

        q_rot, k_rot = NeoXRoPE.apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        q_BHLd = torch.cat((q_rot, q_pass), dim=-1)
        k_BHLd = torch.cat((k_rot, k_pass), dim=-1)

        # print(f"my cos cache is shape: {cos.size()} and tensor:\n {cos}")
        # print(f"my post_rope query, shape: {q_BHLd.size()}, tensor 0,1: {q_BHLd[0][1]}")

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
    """ """

    def __init__(self, config):
        super().__init__()
        self.attention = DecoderAttentionRotary(config)
        self.feed_forward = DecoderFeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)
        self.attention_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)

    def forward(self, x, freqs_cis, mask, parallel_residual=True, use_cache=True):
        # @NOTE :: this i different from BERT*** norm then add vs add then norm
        assert freqs_cis is not None
        # @NOTE PYTHIA USES PARALLEL RESIDUAL STREAMS!!!
        if parallel_residual:
            out = (
                x
                + self.attention(self.attention_norm(x), freqs_cis, mask)
                + self.feed_forward(self.ffn_norm(x))
            )
        else:
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

        self.rotary_pct = 0.25

    def setup_caches(self):
        head_dim = self.config.embed_dim // self.config.num_heads
        dtype = self.output.weight.dtype

        # self.freqs_cis = RoPE.precompute_freqs_cis(
        #     seq_len=self.config.max_pos_embedding,
        #     n_elem=self.config.embed_dim // self.config.num_heads,
        #     base=self.config.rotary_emb_base,
        #     dtype=dtype,
        # )
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
