import torch

from decoder import DecoderWrapper
from configurations import Pythia14MConfig


def test_build_pythia():
    """ """
    pythia = DecoderWrapper(Pythia14MConfig)
    dummy_input = torch.LongTensor(torch.randint(1, 100, (4, 128)))
    out = pythia.forward(dummy_input)
    print(out)
    print(out.size())
    print(pythia)


def test_safetensors():
    """Transfer the pretrained safetensors to"""
    return None


def test_pretrained():
    """ """
    return None


def main():
    test_build_pythia()


if __name__ == "__main__":
    main()
