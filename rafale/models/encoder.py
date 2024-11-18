from dataclasses import dataclass
from torchmetrics.classification import Accuracy
from typing_extensions import Literal

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from torchmetrics import Metric
from torchmetrics.collections import MetricCollection
from torchmetrics.classification.accuracy import BinaryAccuracy

from composer.metrics import LossMetric, MaskedAccuracy
from composer.models import ComposerModel

###############################################################################
#                       SIMPLE BERT-like BUILDING BLOCKS                      #
###############################################################################


# @TODO :: Refactor, improve documentation and add tensor dimension keys for the names


class Embedding(nn.Module):
    """Embeddings

    In addition to the word embedding, BERT uses learned absolute position embeddings. We also have token type embedding for one the BERT pretraining
    objectives.

    Tensor dimension keys:
    - B batch size
    - L sequence length
    - H number of attention heads
    - D embedding dimension
    - d attention head dimension D//H
    - F feedforward dimension
    """

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
            # padding_idx=pad_token_id,  # ROBERTA only?
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
            persistent=False,
        )

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        # move created tensor to same device
        device = input_ids.get_device()

        position_ids = self.position_ids.clone()
        position_ids = position_ids.to(device=device).type_as(input_ids)
        token_type_ids = token_type_ids.to(device=device).type_as(input_ids)

        position_ids = position_ids[:, :seq_length]
        position_ids = position_ids.expand_as(input_ids)

        # we assume absolute positional encoding here like in the original BERT and sum everything up
        W = self.word_embeddings(input_ids)
        P = self.position_embeddings(position_ids)
        T = self.token_type_embeddings(token_type_ids)

        E = W + P + T
        E = self.LayerNorm(E)
        E = self.dropout(E)

        return E


class EncoderSelfAttention(nn.Module):
    """Bidirectional multi-head self attention.

    Tensor dimension keys:
    - B batch size
    - L sequence length
    - H number of attention heads
    - D embedding dimension
    - d attention head dimension D//H
    - F feedforward dimension
    """

    def __init__(self, n_heads, embed_dim, dropout_p=0.1, fast_attn=False):
        super().__init__()
        self.dropout_p = dropout_p
        self.fast_attn = fast_attn
        assert embed_dim % n_heads == 0

        # We assume d_v always equals d_k
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.all_head_size = n_heads * self.head_dim

        # get linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        """"""
        batch_size = q.size(0)
        if not self.training:
            self.dropout_p = 0

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

        attn_output = scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p,
        )

        # concatenate heads and put through final linear layer
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )

        return attn_output


class AttentionModule(nn.Module):
    """the actual block with the output projections"""

    # output
    def __init__(self, n_heads, embed_dim, dropout_p=None, fast_attn=False):
        super().__init__()
        self.self_attn = EncoderSelfAttention(
            n_heads, embed_dim, dropout_p=dropout_p, fast_attn=fast_attn
        )
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        out = self.out(attn_output)
        return out


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
        self.ln = nn.LayerNorm(embed_dim, eps=eps)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, residual):
        x = self.dropout(x)  # @TODO :: make sure this should be here...
        x = self.ln(x + residual)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, eps=None, dropout_p=None):
        super().__init__()

        self.attention = AttentionModule(
            n_heads=n_heads, embed_dim=embed_dim, dropout_p=dropout_p
        )
        self.add_norm_1 = AddNorm(embed_dim, eps=eps, dropout_p=dropout_p)
        self.ff = FeedForward(embed_dim, ff_dim=ff_dim)
        self.add_norm_2 = AddNorm(embed_dim, eps=eps, dropout_p=dropout_p)

    def forward(self, x):
        residual_1 = x
        x = self.attention(x)
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

class ClsHead(nn.Module):
    """Binary or multiclass classfification head for encoders.
    The `pooling` layer (picks out the [CLS] token and a projection
    """
    def __init__(self, embed_dim, num_labels, dropout_p=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()

        #self.dropout = nn.Dropout(dropout_p) # no dropout for now
        self.out = nn.Linear(embed_dim, num_labels)

    def forward(self, x):
        first_token_tensor = x[:, 0] # first [CLS] token only

        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        output = self.out(pooled_output)

        return output

class EncoderWrapper(nn.Module):
    def __init__(self, config, mode : Literal["mlm", "cls"], num_classes=None):
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
        self.token_type_ids = torch.ones(config.max_pos_embedding, dtype=torch.long)

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
        if mode == "mlm":
            self.head = MLMHead(
                embed_dim=config.embed_dim,
                eps=config.layer_norm_eps,
                vocab_size=config.vocab_size,
            )
            # Tie the weights
            self.mlm_head.decoder.weight = self.embedding_layer.word_embeddings.weight
            # @NOTE :: bias are tied too with the HF model
            # no bias for MLM head (?), let's keep it since the HF implementation keeps it as well
            # self.mlm_head.mlm[-1].bias = None
        elif mode == "cls":
            self.head = ClsHead(
                embed_dim=config.embed_dim,
                num_labels=num_classes,
                dropout_p=0.1,
            )
        else:
            raise ValueError(f"Mode: {cls} is not valid.")

    def forward(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = self.token_type_ids
        token_type_ids = token_type_ids.repeat(input_ids.size(0)).view(input_ids.size(0), -1)[:, :input_ids.size(1)]

        x = self.embedding_layer(input_ids, token_type_ids)
        # x = self.encoder_blocks(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)

        return x

class ComposerMLM(ComposerModel):
   """wrapping the encoder for MLM training with CE masked loss"""

   def __init__(self, config, mode) -> None:
       super().__init__()
       self.model = EncoderWrapper(config, mode="mlm")
       self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

       self.train_metrics = MetricCollection(
            [LossMetric(self.ce_loss), MaskedAccuracy()]
        )
       self.eval_metrics = MetricCollection(
            [LossMetric(self.ce_loss), MaskedAccuracy()]
        )

   def loss(self, logits, labels):
        """ """
        # Flatten the logits and labels
        logits = logits.view(
            -1, self.config.vocab_size
        )  # Adjust vocab_size as per your config

        labels = labels.view(-2)

        # Compute and return the loss
        return self.ce_loss(logits, labels)

class ComposerEncoderClassifier(ComposerModel):
   """wrapping the encoder for the trainer"""

   def __init__(self, config, num_classes: int, mode: str = "cls") -> None:
       super().__init__()
       self.model = EncoderWrapper(config, mode=mode, num_classes=num_classes)
       self.ce_loss = nn.CrossEntropyLoss()

       # train as if multiclass for simplicity, works for binary/multiclassification setups
       self.train_metrics = MetricCollection(
            [LossMetric(self.ce_loss), Accuracy(task="multiclass", num_classes=num_classes)]
        )
       self.eval_metrics = MetricCollection(
            [LossMetric(self.ce_loss), Accuracy(task="multiclass", num_classes=2)]
        )

   def forward(self, batch):  # batch is the output of the dataloader
        """batch is a dict with "input_ids" key, model also takes past_kv"""
        # specify how batches are passed through the model
        return self.model(batch)

   def eval_forward(self, batch, outputs=False):
        outputs = self.model(batch)
        return outputs

   def update_metric(self, batch, outputs, metric) -> None:
        labels = batch["labels"]
        metric.update(outputs.view(-1, 2), labels.view(-1))

   def get_metrics(self, is_train=False) -> dict[str, Metric]:
        # defines which metrics to use in each phase of training
        # return self.train_metrics if is_train else self.eval_metrics
        return self.train_metrics if is_train else self.eval_metrics

   def loss(self, outputs, batch):
       labels = batch["labels"]
       return self.ce_loss(outputs.view(-1, 2), labels.view(-1))
