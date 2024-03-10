import argparse
import torch
import torch.nn.functional as F

# from encoder import EncoderWrapper
from transformers import AutoTokenizer, AutoModelForMaskedLM


# helpers
def list_params(model):
    for k, v in model.items():
        print(k)


def get_name_params(model):
    all_params = {}
    for name, param in model.named_parameters():
        all_params[name] = param

    return all_params


def convert_bert_params_dict(target, source):
    conversion = [
        ("bert.embeddings", "embedding_layer"),
        ("bert.encoder.layer", "blocks"),
        ("attention.self", "attention.self_attn"),
        ("attention.output.dense", "attention.out"),
        ("attention.output.LayerNorm", "add_norm_1.ln"),
        ("intermediate.dense", "ff.ff_in"),
        ("output.dense", "ff.ff_out"),
        ("output.LayerNorm", "add_norm_2.ln"),
        ("cls.predictions.bias", "mlm_head.bias"),
        ("cls.predictions.transform.dense", "mlm_head.dense"),
        ("cls.predictions.transform.LayerNorm", "mlm_head.ln"),
        ("cls.predictions.decoder", "mlm_head.decoder"),
    ]

    source_parameters = source.state_dict()

    updated_parameters = {}
    for k, v in source_parameters.items():
        for hf_term, my_term in conversion:
            if hf_term in k:
                k = k.replace(hf_term, my_term)

        updated_parameters[k] = v

    # return updated_parameters
    # assert new_dict.keys() == target.keys(), was Ok but different for the state dict

    # here we transfer weights for all layers
    target.load_state_dict(updated_parameters, strict=True)

    return target


def convert_roberta_params_dict(target, source):
    conversion = [
        ("roberta.embeddings", "embedding_layer"),
        ("roberta.encoder.layer", "blocks"),
        ("attention.self", "mha"),
        ("attention.output.dense", "mha.out"),
        ("attention.output.LayerNorm", "add_norm_1.ln"),
        ("intermediate.dense", "ff.ff_in"),
        ("output.dense", "ff.ff_out"),
        ("output.LayerNorm", "add_norm_2.ln"),
        ("lm_head.bias", "mlm_head.bias"),
        ("lm_head.dense", "mlm_head.dense"),
        ("lm_head.layer_norm", "mlm_head.ln"),
        ("lm_head.decoder", "mlm_head.decoder"),
    ]

    source_parameters = source.state_dict()

    updated_parameters = {}
    for k, v in source_parameters.items():
        for hf_term, my_term in conversion:
            if hf_term in k:
                k = k.replace(hf_term, my_term)

        updated_parameters[k] = v

    # return updated_parameters
    # assert new_dict.keys() == target.keys(), was Ok but different for the state dict

    # here we transfer weights for all layers
    target.load_state_dict(updated_parameters, strict=True)

    return target


def test_conversion():
    """Convert the weights modify the dictionary for the blank and choice tokens, export the resulting tokenizer and
    model checkpoints into assets
    """
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    input_tokens = tokenizer("paris is the [MASK] of France.", return_tensors="pt")
    hf_model = AutoModelForMaskedLM.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    extra_tokens = [
        "<BLANK>",
        "<A>",
        "<B>",
        "<C>",
        "<D>",
        "<E>",
        "<F>",
        "<G>",
        "<H>",
        "<I>",
        "<J>",
        "<K>",
        "<L>",
    ]

    # check if the tokens are already in the vocabulary
    extra_tokens = set(extra_tokens) - set(tokenizer.vocab.keys())

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(extra_tokens))

    # add new, random embeddings for the new tokens
    hf_model.resize_token_embeddings(len(tokenizer))

    cfg = Config
    slam = SlamEncoder(cfg)
    slam = convert_bert_params_dict(slam, hf_model)

    def get_test_preds(model, sample, tokenizer, hf=False):
        model.eval()
        output = model(**sample)

        if hf:
            output = output.logits

        probs = F.softmax(output, dim=-1)
        tokens = torch.argmax(probs, dim=-1)
        sequence = tokenizer.batch_decode(tokens)
        print(tokens)

        return (
            output,
            tokens,
            probs,
            sequence,
        )

    _, _, _, hf_tokens = get_test_preds(hf_model, input_tokens, tokenizer, hf=True)

    _, _, _, my_tokens = get_test_preds(slam, input_tokens, tokenizer)
    # the output logits are different, however the output tokens predicted seem to be almost always the same
    print(f"my implementation: {my_tokens}\n hf implementation: {hf_tokens}")


def add_new_tokens(model, tokenizer, token_list):
    token_list = set(token_list) - set(tokenizer.vocab.keys())

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(token_list))

    # add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def convert_and_save(dump_dir):
    """Convert the weights modify the dictionary for the blank and choice tokens, export the resulting tokenizer and
    model checkpoints into assets
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    hf_model = AutoModelForMaskedLM.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )

    cfg = Config
    slam = SlamEncoder(cfg)
    slam = convert_bert_params_dict(slam, hf_model)

    # save model
    tokenizer.save_pretrained(f"{dump_dir}/tokenizer/")
    torch.save(slam.state_dict(), f"{dump_dir}/model.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Convert bert weights and tokenizer for Slamminnnnn"
    )

    # Adding arguments
    parser.add_argument(
        "--dump_dir",
        required=True,
        type=str,
        help="Where to put the files",
    )

    # Parse arguments
    args = parser.parse_args()
    convert_and_save(args.dump_dir)


if __name__ == "__main__":
    main()
