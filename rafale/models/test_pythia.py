import torch

from safetensors import safe_open

from decoder import DecoderWrapper
from configurations import Pythia14MConfig
from convert_hf_weights import convert_pythia_params_dict


def test_build_pythia():
    """ """
    pythia = DecoderWrapper(Pythia14MConfig)
    dummy_input = torch.LongTensor(torch.randint(1, 100, (4, 128)))
    out = pythia.forward(dummy_input)
    print(out)
    print(out.size())
    print(pythia)


def test_safetensors():
    """Transfer the pretrained safetensors to rafale model"""
    with safe_open("pythia14m.safetensors", framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    pythia_pt = convert_pythia_params_dict(pythia, tensors)

    return None


def test_pretrained():
    """ """
    return None


def main():
    test_build_pythia()


if __name__ == "__main__":
    main()
