import torch
import numpy as np

from safetensors import safe_open

# from decoder import DecoderWrapper
from configurations import Pythia14MConfig
from convert_hf_weights import convert_pythia_params_dict


from transformers import AutoTokenizer


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

    input_str = "Hello World from pythia!"

    tokens = tokenizer(input_str, return_tensors="pt")
    with torch.no_grad():
        hf_out = hf_gpt(tokens["input_ids"])["logits"].detach().numpy()
        rafale_out = rafale_gpt(tokens["input_ids"]).detach().numpy()

    tol = 1e-05
    print(f"checking outputs at atol/rtol {tol}")

    np.testing.assert_allclose(rafale_out, hf_out, rtol=tol, atol=tol)

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


def iterative_debug(rafale_model, hf_model, tokenizer=None, layer=0):
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
    rafale_model.layers[layer].attention_norm.register_forward_hook(
        get_rafale_activation("attn_norm")
    )
    hf_model.gpt_neox.layers[layer].input_layernorm.register_forward_hook(
        get_hf_activation("attn_norm")
    )

    rafale_model.layers[layer].attention_norm.register_forward_hook(
        get_rafale_input_activation("input_attn_norm")
    )
    hf_model.gpt_neox.layers[layer].input_layernorm.register_forward_hook(
        get_hf_input_activation("input_attn_norm")
    )

    # attention projection query_key_values (pre RoPE)
    rafale_model.layers[layer].attention.query_key_value.register_forward_hook(
        get_rafale_activation("attn_inproj")
    )

    hf_model.gpt_neox.layers[layer].attention.query_key_value.register_forward_hook(
        get_hf_activation("attn_inproj")
    )

    # out proj @BUG :: this is where it breaks**
    rafale_model.layers[layer].attention.dense.register_forward_hook(
        get_rafale_activation("attn_dense")
    )

    hf_model.gpt_neox.layers[layer].attention.dense.register_forward_hook(
        get_hf_activation("attn_dense")
    )

    # INPUT check before attention dense* (if this fails then RoPE is probably the problem...)
    rafale_model.layers[layer].attention.dense.register_forward_hook(
        get_rafale_input_activation("attn_dense")
    )
    hf_model.gpt_neox.layers[layer].attention.dense.register_forward_hook(
        get_hf_input_activation("attn_dense")
    )

    # feed forward out
    rafale_model.layers[layer].feed_forward.ff_out.register_forward_hook(
        get_rafale_activation("ffout")
    )
    hf_model.gpt_neox.layers[layer].mlp.dense_4h_to_h.register_forward_hook(
        get_hf_activation("ffout")
    )

    # hf_model.gpt_neox.layers[0].attention(tensor)

    input_str = "Hello World from pythia!"
    tokens = tokenizer(input_str, return_tensors="pt")

    hf_model.eval()
    rafale_model.eval()
    print(f"Dropout p should be 0: {rafale_model.layers[layer].attention.dropout_p}")

    with torch.no_grad():
        hf_out = hf_model(tokens["input_ids"])["logits"].detach().numpy()
        rafale_out = rafale_model(tokens["input_ids"]).detach().numpy()

    # embedding ###################################################################
    np.testing.assert_allclose(
        rafale_activation["input_embeddings"].numpy(),
        hf_activation["input_embeddings"].numpy(),
        rtol=1e-04,
        atol=1e-04,
    )
    print(f"Testing layer {layer}")
    print(f"✅ embeddings OK!")

    # PRE-ATTENTION NORM ######################################################
    try:
        np.testing.assert_allclose(
            rafale_input_activation["input_attn_norm"].numpy(),
            hf_input_activation["input_attn_norm"].numpy(),
            rtol=1e-04,
            atol=1e-04,
        )
        print(f"✅ INPUTS of pre-attention norm OK!")
    except:
        print("⚠️ INPUTS pre-attention norm difference")

    try:
        np.testing.assert_allclose(
            rafale_activation["attn_norm"].numpy(),
            hf_activation["attn_norm"].numpy(),
            rtol=1e-04,
            atol=1e-04,
        )
        print(f"✅ pre-attention norm OK!")
    except:
        print("⚠️ pre-attention norm difference")

    # LINEAR PROJECTION FOR ATTENTION ######################################################

    try:
        np.testing.assert_allclose(
            rafale_activation["attn_inproj"].numpy(),
            hf_activation["attn_inproj"].numpy(),
            rtol=1e-04,
            atol=1e-04,
        )
        print(f"✅ attention in-projection OK!")

    except:
        print(f"⚠️ attention in-projection difference")

    # INPUTS OF ATTENTION DENSE LAYER  ########################################
    try:
        np.testing.assert_allclose(
            rafale_input_activation["attn_dense"].numpy(),
            hf_input_activation["attn_dense"].numpy(),
            rtol=1e-04,
            atol=1e-04,
        )
    except:
        print("⚠️ outputs of attention difference")

    print(f"✅ post-attention OK!")

    # OUTPUT OF ATTENTION DENSE LAYER  ########################################
    try:
        np.testing.assert_allclose(
            rafale_activation["attn_dense"].numpy(),
            hf_activation["attn_dense"].numpy(),
            rtol=1e-04,
            atol=1e-04,
        )
        print(f"✅ attention out dense OK!")
    except:
        print("⚠️ attention out dense difference")

    try:
        np.testing.assert_allclose(
            rafale_activation["ffout"].numpy(),
            hf_activation["ffout"].numpy(),
            rtol=1e-04,
            atol=1e-04,
        )
        print(f"✅ feedforward out dense OK!")
    except:
        print("⚠️ ff out dense difference")


def test_incremental():
    """ """
    torch.manual_seed(0)
    np.random.seed(0)

    hf_pythia = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-14m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")

    rafale_pythia = test_build_pythia()  # check forward pass OK
    print(f"✅ model initialization OK!")

    rafale_pythia = test_safetensors(rafale_pythia)
    print(f"✅ model weight transfers OK!")

    test_eval(rafale_pythia, tokenizer)
    print(f"✅ model determinism OK!")

    for i in range(6):
        iterative_debug(rafale_pythia, hf_model=hf_pythia, tokenizer=tokenizer, layer=i)

    rafale_outputs, hf_outputs = test_pretrained(
        rafale_pythia, hf_gpt=hf_pythia, tokenizer=tokenizer
    )
    print(f"✅ full model outputs OK!")

    return rafale_outputs, hf_outputs


if __name__ == "__main__":
    main()
