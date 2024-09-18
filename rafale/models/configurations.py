import requests
import os

from dataclasses import dataclass

from safetensors import safe_open

from ..caches import MODEL_CACHE_DIR

"""
to simplify model loading add a configuration for the pre-trained weight loading using safetensors instead of loading
the full model.
> then save to a folder named ".pretrained/" in this directory
"""


def download_file(url, save_path):
    """
    Downloads a file from the specified URL to the given save path.

    :param url: The URL of the file to download.
    :param save_path: The local path where the file will be saved.
    """
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Check for HTTP errors
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kilobyte
            progress = 0

            with open(save_path, "wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress += len(data)
                    print(
                        f"Downloaded {progress} of {total_size} bytes ({(progress/total_size)*100:.2f}%)",
                        end="\r",
                    )

        print(f"\nDownload completed successfully! File saved to: {save_path}")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # HTTP error
    except Exception as err:
        print(f"An error occurred: {err}")  # Other errors


def load_safetensors(rafale_model, model_config):
    """Transfer the pretrained safetensors to rafale model"""
    tensors = {}

    safetensors_path = os.path.join(MODEL_CACHE_DIR, model_config.name + ".safetensors")

    if os.path.isfile(safetensors_path):
        pass
    else:
        download_file(model_config.safetensors_url, safetensors_path)

    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    rafale_model = model_config.convert_params_dict(rafale_model, tensors)

    return rafale_model


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
class BertTinyConfig:
    embed_dim: int = 128
    vocab_size: int = 30522  # could usage would be to 30522 + num_extra_tokens
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    num_heads: int = 2
    ff_dim: int = 512
    max_pos_embedding: int = 512
    layer_norm_eps: float = 1e-12
    num_blocks: int = 2
    pad_token_id: int = 0
    num_token_type: int = 2
    fast_attention: bool = False  # use xformers (todo: add FlashAttention2)


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
    name: str = "pythia14m"
    safetensors_url: str = (
        "https://huggingface.co/EleutherAI/pythia-14m/resolve/main/model.safetensors"
    )

    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 512
    hidden_act: str = "gelu"
    max_pos_embedding: int = 2048
    vocab_size: int = 50304

    parallel_residual: bool = True

    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1

    layer_norm_eps: float = 1e-05
    num_blocks: int = 6
    # pad_token_id: int = -100

    bos_token_id: int = 0
    eos_token_id: int = 0
    fast_attention: bool = False

    rotary_emb_base: int = 10000
    rotary_pct: float = 0.25

    tie_word_embeddings: bool = False

    @classmethod
    def convert_params_dict(cls, target, source):
        """
        Source safetensors dict to our rafale model class.
        """
        # not needed for our implementation
        unused = ["rotary_emb.inv_freq", "masked_bias", "attention.bias"]
        for k, v in list(source.items()):
            if True in [x in k for x in unused]:
                del source[k]

        conversion = [
            ("gpt_neox.embed_in", "token_embeddings.input_embeddings"),
            ("gpt_neox.layers", "layers"),
            ("input_layernorm", "attention_norm"),
            ("post_attention_layernorm", "ffn_norm"),
            ("mlp", "feed_forward"),
            ("dense_4h_to_h", "ff_out"),
            ("dense_h_to_4h", "ff_in"),
            ("embed_out", "output"),
            ("gpt_neox.final_layer_norm", "final_norm"),
        ]

        updated_parameters = {}
        for k, v in source.items():
            for hf_term, my_term in conversion:
                if hf_term in k:
                    k = k.replace(hf_term, my_term)

            updated_parameters[k] = v

        # here we transfer weights for all layers
        target.load_state_dict(updated_parameters, strict=True)

        return target
