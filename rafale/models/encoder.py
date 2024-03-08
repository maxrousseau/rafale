from dataclasses import dataclass

import torch
import torch.nn.functional as F
import xformers.ops as xops
from torch import nn
from torch.nn.functional import scaled_dot_product_attention


class Embedding(nn.Module):
    """embeddings from word, absolute/fixe position (and token_type embedding?)"""

    def __init__(
        self,
        vocab_size=None,
        hidden_size=None,
        pad_token_id=None,
        max_sequence_length=512,
        num_token_type=None,  # technically should be 2, in HF they use type_vocab_size
        layer_norm_eps=None,
        hidden_dropout_prob=None,
    ):
        super().__init__()
        # nn.Embedding is just a lookup table,
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            num_token_type, hidden_size
        )  # NOTE :: these are actually the segment embeddings
        # from the original BERT paper... maybe rename?
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # not considered as model parameter
        self.register_buffer(
            "position_ids",
            torch.arange(max_sequence_length).expand((1, -1)),
        )

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)

        position_ids = self.position_ids[:seq_length]
        position_ids = position_ids.expand_as(input_ids)

        # we assume absolute positional encoding here like in the original BERT and sum everything up
        W = self.word_embeddings(input_ids)
        P = self.position_embeddings(position_ids)
        T = self.token_type_embeddings(token_type_ids)

        E = W + P + T
        E = self.LayerNorm(E)
        E = self.dropout(E)

        return E


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, dropout_p=None, fast_attn=False):
        "uses xformers memory effeicient attention"
        super().__init__()
        self.dropout_p = dropout_p
        self.fast_attn = fast_attn
        assert embed_dim % n_heads == 0
        # We assume d_v always equals d_k
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.all_head_size = n_heads * self.head_dim

        # @TODO get linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # output
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        # @HERE TODO
        """
        so input q, k, v are essentially the same tensor since we're doing self attention but the idea here is that this
        same implementation could be used for cross attention.

        in self-attention they have dimension [batch_size, sequence_length, embedding_dimension] so for bert that would
        be like [4, 512, 768] for example.

        each attention head will handle part of the embedding dimensions (wow I didn't know that and don't fully
        understand why...). So this is why we want to have embed_dim % n_head == 0.

        (1) we use view to reshape the tensor into shape [batch_size, seq_length, n_head, head_embed] --> .view(batch_size,
        -1, self.num_heads, self.head_dim)
        (2) then we transpose seq_length and n_head to parrellalize the computations during the attention computations
        --> .transpose(1, 2)


        ## Summary of Shape Changes
        Input: [batch_size, seq_length, embed_dim]
        Post Linear Layer: [batch_size, seq_length, embed_dim] (same shape, but transformed)
        View for Heads: [batch_size, seq_length, num_heads, head_dim]
        Transpose for Heads: [batch_size, num_heads, seq_length, head_dim]

        ## after having applied attention
        We receive a tensor of shape [batch_size, num_heads, seq_length, head_dim] (same as before)
        Now we want to get back to our original embedding and sequence shape so first we swap back num_head and
        seq_length with --> .transpose(1,2)
        Then we want to aggregate our head_dim to have our full embedding space back up together again with -->
        .view(batch_size, -1, self.embed_dim)
        and we get shape [batch_size, seq_length, embed_dim] at the end

        """

        batch_size = q.size(0)
        if not self.training:
            self.dropout = 0
            print("model not training, attention dropout is 0")

        # check transformation again here....
        q = (
            self.query(q)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(k)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(v)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        # check for flash-attn-2? optional
        if self.fast_attn:
            attn_output = xops.memory_efficient_attention(
                q,
                k,
                v,
                p=self.dropout_p,
            )

        else:
            attn_output = scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p,
            )

        # Concatenate heads and put through final linear layer
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )
        return self.out(attn_output)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.ff_in = nn.Linear(embed_dim, ff_dim)
        self.gelu = nn.GELU()
        self.ff_out = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = self.ff_in(x)
        x = self.gelu(x)
        x = self.ff_out(x)
        return x


class AddNorm(nn.Module):
    def __init__(self, embed_dim, eps=None, dropout_p=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x, residual):
        x = self.dropout(x)
        x = self.ln(x + residual)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, eps=None, dropout_p=None):
        super().__init__()

        self.mha = MultiHeadAttention(
            n_heads=n_heads, embed_dim=embed_dim, dropout_p=dropout_p
        )
        self.add_norm_1 = AddNorm(embed_dim, eps=eps, dropout_p=dropout_p)
        self.ff = FeedForward(embed_dim, ff_dim=ff_dim)
        self.add_norm_2 = AddNorm(embed_dim, eps=eps, dropout_p=dropout_p)

    def forward(self, x):
        residual_1 = x
        x = self.mha(x, x, x)
        x = self.add_norm_1(x, residual_1)

        residual_2 = x
        x = self.ff(x)
        x = self.add_norm_2(x, residual_2)

        return x


class MLMHead(nn.Module):
    def __init__(self, embed_dim, vocab_size, eps=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(embed_dim, eps=eps)

        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.dense(x)
        x = self.gelu(x)
        x = self.ln(x)
        x = self.decoder(x)

        return x


class SlamEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = Embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.embed_dim,
            pad_token_id=config.pad_token_id,
            max_sequence_length=config.max_pos_embedding,
            num_token_type=config.num_token_type,  # technically should be 2, in HF they use type_vocab_size
            layer_norm_eps=config.layer_norm_eps,
            hidden_dropout_prob=config.hidden_dropout,
        )

        self.blocks = nn.ModuleList()
        for i in range(config.num_blocks):
            self.blocks.append(
                EncoderBlock(
                    config.embed_dim,
                    config.num_heads,
                    config.ff_dim,
                    eps=config.layer_norm_eps,
                    dropout_p=config.hidden_dropout,
                )
            )

        self.mlm_head = MLMHead(
            embed_dim=config.embed_dim,
            eps=config.layer_norm_eps,
            vocab_size=config.vocab_size,
        )

        # Tie the weights
        self.mlm_head.decoder.weight = self.embedding_layer.word_embeddings.weight
        # @NOTE :: bias are tied too with the HF model

    # no bias for MLM head (?), let's keep it since the HF implementation keeps it as well
    # self.mlm_head.mlm[-1].bias = None

    def compute_decoupled_label_loss(self, logits, labels, permutations, tokenizer):
        """ """
        batch_size = logits.size(0)
        F.softmax(logits, dim=-1)
        letters = list(map(chr, range(97, 123)))

        # Define the loss functions
        # same used as adapet

        nn.BCELoss(reduction="none")
        nn.BCEWithLogitsLoss(reduction="none")

        losses = []

        for i in range(batch_size):
            num_labels = permutations[i]
            # get all the relevant ids (choices) from the sample
            relevant_ids = tokenizer.convert_tokens_to_ids(letters[:num_labels])

            # the token id of the positive label of the example
            example_label = labels[i]
            example_label_id = example_label[example_label != -100].item()

            if len(relevant_ids) == 1:
                # caveat if example only has one choice...
                example_bad_ids = []
            else:
                relevant_ids.remove(example_label_id)
                example_bad_ids = relevant_ids

            # get the logit predictions from the mask token
            l = logits[i]
            # l = probabilities[i]
            mask_token_logits = l[labels[i] != -100]
            mask_token_logits = torch.flatten(mask_token_logits)

            indices = [x for x in range(l.size(1)) if x not in relevant_ids]
            non_choice_logits = torch.index_select(
                mask_token_logits, 0, torch.tensor(indices)
            )

            # probability logits for the positive label
            positive_prediction = mask_token_logits[example_label_id]

            negative_losses = []
            negative_label = torch.zeros([])
            # probability logits for the negative labels
            for idx in example_bad_ids:
                # negative_predictions.append(mask_token_logits[idx])
                negative_losses.append(
                    self.bcel_loss(mask_token_logits[idx], negative_label)
                )

            nulltoken_labels = torch.zeros(len(indices))  # device="cuda:0"
            positive_labels = torch.ones([])

            # mean of bad labels bcel
            nulltoken_label_loss = torch.mean(
                self.bcel_loss(non_choice_logits, nulltoken_labels)
            )
            negative_label_loss = torch.sum(torch.stack(negative_losses))
            positive_label_loss = self.bcel_loss(positive_prediction, positive_labels)

            losses.append(
                torch.sum(
                    torch.stack(
                        [
                            positive_label_loss.view(1),
                            nulltoken_label_loss.view(1),
                            negative_label_loss.view(1),
                        ]
                    )
                )
            )
        return torch.mean(torch.stack(losses))

    def compute_loss(self, logits, labels):
        """ """
        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

        # Flatten the logits and labels
        logits = logits.view(
            -1, self.config.vocab_size
        )  # Adjust vocab_size as per your config
        labels = labels.view(-1)

        # Compute and return the loss
        return ce_loss(logits, labels)

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        token_type_ids = kwargs["token_type_ids"]
        x = self.embedding_layer(input_ids, token_type_ids)
        for block in self.blocks:
            x = block(x)
        x = self.mlm_head(x)

        return x


# DELETE THIS
def get_tokens_from_logits(logits, tokenizer=None):
    """
    return the prediced tokens for all of the inputs
    """
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Get the predicted token IDs
    predicted_token_ids = torch.argmax(probabilities, dim=-1)

    predicted_tokens = [
        tokenizer.convert_ids_to_tokens(seq.numpy())
        for seq in torch.unbind(predicted_token_ids, dim=0)
    ]
    return predicted_tokens


@dataclass
class Config:
    # defaults >> "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
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
