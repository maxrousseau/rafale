import os

from datapipe import WikiMLMPipe
from encoder import EncoderWrapper, BertConfig
from roberta import RobertaConfig, RobertaMLM
from convert_hf_weights import convert_bert_params_dict, convert_roberta_params_dict

from transformers import AutoTokenizer, AutoModelForMaskedLM

import torch

import numpy as np
import random

torch.set_deterministic_debug_mode(1)
torch.use_deterministic_algorithms(True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# dump modeling debugging code here...
# @TODO :: call from cli, layer by layer check, add proper logging


# roberta
def test_roberta():
    # roberta base
    roberta_cfg = RobertaConfig()
    r_roberta = RobertaMLM(roberta_cfg)

    roberta_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    hf_roberta = AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-base")

    r_roberta = convert_roberta_params_dict(r_roberta, hf_roberta)

    args = {
        "path": "~/code/data/enwiki1m",
        "truncation": True,
        "max_sequence_length": 128,
        "shuffle_train": False,
        "batch_size": 1,
        "padding": "max_length",
        "tokenizer": roberta_tokenizer,
    }

    wikipipe = WikiMLMPipe(**args)
    dl = wikipipe()

    batch = next(iter(dl["train"]))

    with torch.no_grad():
        r_roberta.eval()
        hf_roberta.eval()
        r_output = r_roberta(**batch)
        r_output2 = r_roberta(**batch)
        hf_output = hf_roberta(**batch)
        hf_output2 = hf_roberta(**batch)
        np.testing.assert_allclose(
            hf_output.logits.detach().numpy(),
            hf_output2.logits.detach().numpy(),
            atol=5e-4,
            rtol=5e-4,
        )
        print("hf deterministic")
        np.testing.assert_allclose(
            r_output.detach().numpy(), r_output2.detach().numpy(), atol=5e-4, rtol=5e-4
        )
        print("rafale deterministic")
        np.testing.assert_allclose(
            r_output.detach().numpy(),
            hf_output.logits.detach().numpy(),
            atol=5e-4,
            rtol=5e-4,
        )


def test_bert():
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    hf_bert = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    bert_cfg = BertConfig()
    r_bert = EncoderWrapper(bert_cfg)
    r_bert = convert_bert_params_dict(r_bert, hf_bert)

    args = {
        "path": "~/code/data/enwiki1m",
        "truncation": True,
        "max_sequence_length": 128,
        "shuffle_train": False,
        "batch_size": 1,
        "padding": "max_length",
        "tokenizer": bert_tokenizer,
    }

    wikipipe = WikiMLMPipe(**args)
    dl = wikipipe()

    batch = next(iter(dl["train"]))

    with torch.no_grad():
        r_bert.eval()
        hf_bert.eval()
        r_output = r_bert(**batch)
        r_output2 = r_bert(**batch)
        hf_output = hf_bert(**batch)
        hf_output2 = hf_bert(**batch)
        np.testing.assert_allclose(
            hf_output.logits.detach().numpy(),
            hf_output2.logits.detach().numpy(),
            atol=5e-4,
            rtol=5e-4,
        )
        print("hf deterministic")
        np.testing.assert_allclose(
            r_output.detach().numpy(), r_output2.detach().numpy(), atol=5e-4, rtol=5e-4
        )
        print("rafale deterministic")
        np.testing.assert_allclose(
            r_output.detach().numpy(),
            hf_output.logits.detach().numpy(),
            atol=5e-4,
            rtol=5e-4,
        )
