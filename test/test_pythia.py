import torch
import numpy as np

from safetensors import safe_open

from rafale.models.decoder import DecoderWrapper
from rafale.models.configurations import Pythia14MConfig, load_safetensors

from transformers import AutoTokenizer, GPTNeoXForCausalLM


def test_layer_and_outputs(rafale_model, hf_model, tokenizer, layer=0, tol=1e-05):
    """
    # @NOTE :: this currently on evaluates with KV cache enabled, write the test to run this function without the KV cache
    """
    hf_activation = {}
    hf_input_activation = {}
    rafale_activation = {}
    rafale_input_activation = {}

    # tuple of shape num_layers, 2 (keys, values), tensor BHLd
    # make a fake kv-cache of length 4
    kv_cache = []
    n_layers = 6
    cache_len = 4
    n_heads = 4
    head_dim = 32
    for i in range(n_layers):
        k = torch.randn(1, n_heads, cache_len, 32)
        v = torch.randn(1, n_heads, cache_len, 32)
        kv_cache.append((k, v))

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

    # out proj
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

    with torch.no_grad():
        # hf_out = hf_model(tokens["input_ids"])["logits"].detach().numpy()

        hf_out = hf_model(tokens["input_ids"], use_cache=True, past_key_values=kv_cache)
        hf_out = hf_out["logits"].detach().numpy()

        # rafale_out = rafale_model(tokens)[0].detach().numpy()

        rafale_out = rafale_model(tokens, past_kv_cache=kv_cache)[0].detach().numpy()

    print(f"Dropout p should be 0: {rafale_model.layers[layer].attention.dropout_p}")
    print(f"Testing layer {layer}")
    # EMBEDDING  ###################################################################
    try:
        np.testing.assert_allclose(
            rafale_activation["input_embeddings"].numpy(),
            hf_activation["input_embeddings"].numpy(),
            rtol=tol,
            atol=tol,
        )

        print(f"✅ embeddings OK!")
    except:
        print("⚠️ Embedding difference!")

    # PRE-ATTENTION NORM ######################################################
    try:
        np.testing.assert_allclose(
            rafale_input_activation["input_attn_norm"].numpy(),
            hf_input_activation["input_attn_norm"].numpy(),
            rtol=tol,
            atol=tol,
        )
        print(f"✅ INPUTS of pre-attention norm OK!")
    except:
        print("⚠️ INPUTS pre-attention norm difference")

    try:
        np.testing.assert_allclose(
            rafale_activation["attn_norm"].numpy(),
            hf_activation["attn_norm"].numpy(),
            rtol=tol,
            atol=tol,
        )
        print(f"✅ pre-attention norm OK!")
    except:
        print("⚠️ pre-attention norm difference")

    # LINEAR PROJECTION FOR ATTENTION ######################################################

    try:
        np.testing.assert_allclose(
            rafale_activation["attn_inproj"].numpy(),
            hf_activation["attn_inproj"].numpy(),
            rtol=tol,
            atol=tol,
        )
        print(f"✅ attention in-projection OK!")

    except:
        print(f"⚠️ attention in-projection difference")

    # INPUTS OF ATTENTION DENSE LAYER  ########################################
    try:
        np.testing.assert_allclose(
            rafale_input_activation["attn_dense"].numpy(),
            hf_input_activation["attn_dense"].numpy(),
            rtol=tol,
            atol=tol,
        )
        print(f"✅ inputs of attention dense OK")

    except:
        r = rafale_input_activation["attn_dense"].numpy()
        h = hf_input_activation["attn_dense"].numpy()

        print(r.shape)
        print(h.shape)
        print("⚠️ inputs of attention dense difference")

    np.testing.assert_allclose(
        rafale_input_activation["attn_dense"].numpy(),
        hf_input_activation["attn_dense"].numpy(),
        rtol=tol,
        atol=tol,
    )

    # OUTPUT OF ATTENTION DENSE LAYER  ########################################
    try:
        np.testing.assert_allclose(
            rafale_activation["attn_dense"].numpy(),
            hf_activation["attn_dense"].numpy(),
            rtol=tol,
            atol=tol,
        )
        print(f"✅ attention out dense OK!")
    except:
        print("⚠️ attention out dense difference")

    try:
        np.testing.assert_allclose(
            rafale_activation["ffout"].numpy(),
            hf_activation["ffout"].numpy(),
            rtol=tol,
            atol=tol,
        )
        print(f"✅ feedforward out dense OK!")
    except:
        print("⚠️ ff out dense difference")

    # Final model output  ########################################
    try:
        np.testing.assert_allclose(rafale_out, hf_out, rtol=tol, atol=tol)
        print(f"🎉 Model outputs match reference implementation!")
    except:
        print("❌ Model outputs do not match")


def main():
    """ """
    torch.manual_seed(0)
    np.random.seed(0)

    hf_pythia = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-14m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")

    rafale_pythia = DecoderWrapper(Pythia14MConfig)  # check forward pass OK
    rafale_pythia = load_safetensors(rafale_pythia, Pythia14MConfig)

    test_layer_and_outputs(rafale_pythia, hf_pythia, tokenizer)


if __name__ == "__main__":
    main()
