#!/usr/bin/env python
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

###############################################################################
#                 simple implementation of GPT building blocks                #
###############################################################################


class DecoderEmbedding(nn.Module):
    """simply an input projection of the tokens here, since rotary position encodings are used, makes things simpler"""

    def __init__(
        self,
        vocab_size=None,
        hidden_size=None,
        hidden_dropout_prob=None,
    ):
        super().__init__()
        # nn.Embedding is just a lookup table,
        self.input_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            # padding_idx=pad_token_id, # not specified for causal GPTNeoX... ? @TODO :: i think padding is handled by
            # the attention mask...
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.input_embeddings(x)
        return self.dropout(x)

        #        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) #* @NOTE :: interesting no layer norm in the


#        pythia reference HF implementation, is this a thing for all causal decoders? Ans: since they use input
#        layernorm and post attn layer norm this isn't necessary here


class RotaryCache(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


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

    def __init__(self):
        super().__init__()

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = config.embed_dim

        # set bias to True or False (@TODO)
        self.query_key_value = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(p=config.attention_dropout)

    def _split_heads(self, tensor: Tensor):
        """
        Splits hidden dim into attn_head_size and num_attention_heads

        # input tensor: [bs, seq_len, hidden_size]
        # returns: [bs, num_attention_heads, seq_len, attn_head_size]
        """

        return tensor.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

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
            self.num_attention_heads * self.attn_head_size,
        )
        # -> [bs, seq_len, hidden_size]
        return tensor

    def forward(self, x_BLD, freq_cis, mask):
        # projection
        q_BLD, k_BLD, v_BLD = self.query_key_value(x_BLD).split(self.embed_dim, dim=-1)

        q_BHLd = self._split_heads(q_BLD)
        k_BHLd = self._split_heads(k_BLD)
        v_BHLd = self._split_heads(v_BLD)

        # apply rotary embeddings
        q_BHLd = apply_rotary_emb(q_BHLd, freqs_cis)
        k_BHLd = apply_rotary_emb(k_BHLd, freqs_cis)

        # compute attention
        attn_out_BLHd = flex_attention(query, key, value, block_mask=mask)

        # @HERE!!
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

    def __init__(self):
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

    def __init__(self):
        super().__init__()
        self.attention = DecoderAttentionRotary(config)
        self.feed_forward = DecoderFeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.dim, config.norm_eps)
        self.attention_norm = nn.LayerNorm(config.dim, config.norm_eps)

    def forward(self, x, freq_cis, input_pos, mask):
        # @NOTE :: this i different from BERT*** norm then add vs add then norm
        h = x + self.attention(self.attention_norm(x), freq_cis, input_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class DecoderWrapper(nn.Module):
    """
    self explanatory
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rotary_cache = RotaryCache(
            config.embed_dim,
            max_position_embeddings=config.max_pos_embedding,
            base=config.rotary_emb_base,
        )  # rm this

        self.token_embeddings = DecoderEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.embed_dim,
            hidden_dropout_prob=config.hidden_dropout,
        )

        self.layers = nn.ModuleList(
            DecoderBlock(
                hidden_size=config.embed_dim,
                ffn_size=config.ff_dim,
                attn_dropout=config.attention_dropout,
                num_heads=config.num_heads,
            )
            for _ in range(config.num_blocks)
        )
        self.final_norm = nn.LayerNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = config.max_pos_embedding

    def setup_caches(self, max_batch_size, max_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return

        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        dtype = self.output.weight.dtype

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.embed_dim // self.config.num_heads,
            self.config.rotary_emb_base,
            dtype,
        )

        # self.causal_mask = torch.tril(
        #     torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        # )
        self.causal_mask = create_block_mask(
            causal,
            B=None,
            H=None,
            Q_LEN=self.max_seq_length,
            KV_LEN=self.max_seq_length,
        )  # batch and heads will be broadcast

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None):
        # @TODO :: what are the input pos doing here?...
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]  # we crop the excessive length?
        freq_cis = self.freqs_cis[input_pos]

        x = self.token_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freq_cis, mask)
        x = self.final_norm(x)
        logits = self.output(x)
        return logits


# RoPE ###########################################################
# simplest, from gpt-fast
def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


# RoPE ###########################################################
# source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
# @TODO :: read the RoPE paper and break it down to understand what is going on here...
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
