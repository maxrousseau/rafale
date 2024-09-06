import torch
import numpy as np

from safetensors import safe_open

# from decoder import DecoderWrapper
from configurations import Pythia14MConfig
from convert_hf_weights import convert_pythia_params_dict


from transformers import AutoModelForCausalLM, AutoTokenizer


def test_build_pythia():
    """ """
    pythia = DecoderWrapper(Pythia14MConfig)
    dummy_input = torch.LongTensor(torch.randint(1, 100, (4, 128)))

    out = pythia.forward(dummy_input)
    print(out)
    print(out.size())
    print(pythia)

    return pythia


def test_safetensors(rafale_gpt):
    """Transfer the pretrained safetensors to rafale model"""
    tensors = {}
    with safe_open("pythia14m.safetensors", framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    pythia_pt = convert_pythia_params_dict(rafale_gpt, tensors)

    return pythia_pt


def test_pretrained(rafale_gpt, hf_gpt=None, tokenizer=None):
    """make sure to disable all dropout (.eval() will not do this for the scaled_dot_attention dropout_p*)"""

    hf_gpt.eval()
    rafale_gpt.eval()

    for l in rafale_gpt.layers:
        l.attention.dropout_p = 0  # this is bc of the scaled_dot_prod implementation

    input_str = "Hello World from pythia!"

    tokens = tokenizer(input_str, return_tensors="pt")

    with torch.no_grad():
        hf_out = hf_gpt(tokens["input_ids"])["logits"].detach().numpy()
        rafale_out = rafale_gpt(tokens["input_ids"]).detach().numpy()

    np.testing.assert_allclose(rafale_out, hf_out, rtol=1e-05, atol=1e-05)

    return rafale_out, hf_out


# eval works!
def test_eval(rafale_gpt, tokenizer):
    rafale_gpt.eval()

    input_str = "Hello World from pythia!"

    tokens = tokenizer(input_str, return_tensors="pt")

    with torch.no_grad():
        a = rafale_gpt(tokens["input_ids"]).detach().numpy()
        b = rafale_gpt(tokens["input_ids"]).detach().numpy()

    np.testing.assert_allclose(a, b, rtol=1e-05, atol=1e-05)
    print(np.allclose(a, b, rtol=1e-05, atol=1e-05))


def main():
    test_build_pythia()


def test_block(rafale_model, hf_model):
    """ """
    return None


def test_embedding(rafale_model, hf_model, token_ids):
    """ """
    rafale_model.eval()
    hf_model.eval()

    with torch.no_grad():
        r_out = rafale_model.token_embeddings(token_ids)
        hf_out = hf_model.gpt_neox.embed_in(token_ids)

    np.testing.assert_allclose(r_out, hf_out, rtol=1e-05, atol=1e-05)
    print(np.allclose(r_out, hf_out, rtol=1e-05, atol=1e-05))

    return r_out


def test_mlp():
    None


def test_attention(rafale_model, huggingface_model, tensor, causal_mask):
    hf_out = hf_model.gpt_neox.layers[0].attention(tensor)
    rafale_out = rafale_model.layers[0].attention(tensor, rafale_model.freqs_cis, None)
    return None


# does not pass! here is the problem....
# def test_decoder_block(rafale_model, hf_model, embed_array):
#     rafale_model.eval()
#     hf_model.eval()

#     for l in rafale_model.layers:
#         l.attention.dropout_p = 0

#     with torch.no_grad():
#         r_out = rafale_model.layers[0](embed_array, rafale_model.freqs_cis, None)
#         hf_out = hf_model.gpt_neox.layers[0](embed_array)

#     np.testing.assert_allclose(r_out, hf_out[0], rtol=1e-01, atol=1e-01)
#     print(np.allclose(r_out, hf_out[0], rtol=1e-05, atol=1e-05))

#     return r_out


def iterative_debug(rafale_model, hf_model, tokenizer=None):
    """
    save model outputs in dict and compare to locate the issue
    """
    hf_activation = {}
    hf_input_activation = {}
    rafale_activation = {}
    rafale_input_activation = {}

    def get_hf_activation(name):
        def hook(model, input, output):
            hf_activation[name] = output.detach()

        return hook

    def get_hf_input_activation(name):
        def hook(model, _input, output):
            hf_input_activation[name] = _input[0].detach()

        return hook

    def get_rafale_activation(name):
        def hook(model, input, output):
            rafale_activation[name] = output.detach()

        return hook

    def get_rafale_input_activation(name):
        def hook(model, _input, output):
            rafale_input_activation[name] = _input[0].detach()

        return hook

    # embeddings
    rafale_model.token_embeddings.register_forward_hook(
        get_rafale_activation("input_embeddings")
    )
    hf_model.gpt_neox.embed_in.register_forward_hook(
        get_hf_activation("input_embeddings")
    )

    # input layernorm
    rafale_model.layers[0].attention_norm.register_forward_hook(
        get_rafale_activation("attn_norm")
    )
    hf_model.gpt_neox.layers[0].input_layernorm.register_forward_hook(
        get_hf_activation("attn_norm")
    )

    # attention projection query_key_values (pre RoPE)
    rafale_model.layers[0].attention.query_key_value.register_forward_hook(
        get_rafale_activation("attn_inproj")
    )

    hf_model.gpt_neox.layers[0].attention.query_key_value.register_forward_hook(
        get_hf_activation("attn_inproj")
    )

    # out proj @BUG :: this is where it breaks**
    rafale_model.layers[0].attention.dense.register_forward_hook(
        get_rafale_activation("attn_dense")
    )

    hf_model.gpt_neox.layers[0].attention.dense.register_forward_hook(
        get_hf_activation("attn_dense")
    )

    # INPUT check before attention dense* (if this fails then RoPE is probably the problem...)
    rafale_model.layers[0].attention.dense.register_forward_hook(
        get_rafale_input_activation("attn_dense")
    )
    hf_model.gpt_neox.layers[0].attention.dense.register_forward_hook(
        get_hf_input_activation("attn_dense")
    )

    # hf_model.gpt_neox.layers[0].attention(tensor)

    input_str = "Hello World from pythia!"
    tokens = tokenizer(input_str, return_tensors="pt")

    with torch.no_grad():
        hf_out = hf_model(tokens["input_ids"])["logits"].detach().numpy()
        rafale_out = rafale_model(tokens["input_ids"]).detach().numpy()

    np.testing.assert_allclose(
        rafale_activation["input_embeddings"].numpy(),
        hf_activation["input_embeddings"].numpy(),
        rtol=1e-05,
        atol=1e-05,
    )
    print(f"✅ embeddings OK!")

    np.testing.assert_allclose(
        rafale_activation["attn_norm"].numpy(),
        hf_activation["attn_norm"].numpy(),
        rtol=1e-05,
        atol=1e-05,
    )
    print(f"✅ pre-attention norm OK!")

    np.testing.assert_allclose(
        rafale_activation["attn_inproj"].numpy(),
        hf_activation["attn_inproj"].numpy(),
        rtol=1e-05,
        atol=1e-05,
    )
    print(f"✅ attention in-projection OK!")

    # return (
    #     rafale_input_activation["attn_dense"],
    #     hf_input_activation["attn_dense"],
    # )

    np.testing.assert_allclose(
        rafale_input_activation["attn_dense"].numpy()[0][0],
        hf_input_activation["attn_dense"].numpy()[0][0],
        rtol=1e-05,
        atol=1e-05,
    )

    print(f"✅ post-attention OK!")

    np.testing.assert_allclose(
        rafale_activation["attn_dense"].numpy()[0][0],
        hf_activation["attn_dense"].numpy()[0][0],
        rtol=1e-05,
        atol=1e-05,
    )
    print(f"✅ attention out dense OK!")

    # the first token is fine for all (i.e. the only token where RoPE is not applied, then the shit hits the fan, so it
    # absolutely is the RoPE scaling which is fucked... redo implementation following GPTNeoX more closely


def test_incremental():
    """ """
    torch.manual_seed(0)
    np.random.seed(0)

    hf_pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")

    rafale_pythia = test_build_pythia()  # check forward pass OK
    print(f"✅ model initialization OK!")

    rafale_pythia = test_safetensors(rafale_pythia)
    print(f"✅ model weight transfers OK!")

    test_eval(rafale_pythia, tokenizer)
    print(f"✅ model determinism OK!")

    iterative_debug(rafale_pythia, hf_model=hf_pythia, tokenizer=tokenizer)

    # outputs = test_pretrained(rafale_pythia, hf_gpt=hf_pythia, tokenizer=tokenizer)
    # print(f"✅ model outputs OK!")


if __name__ == "__main__":
    main()
