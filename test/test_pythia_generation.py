import torch
import numpy as np

# from safetensors import safe_open
# from tokenizers import Tokenizer

# from rafale.datapipe import InferenceDatapipeline
# from rafale.models.decoding_strategies import greedy_decode
# from rafale.models.decoder import DecoderWrapper
# from rafale.models.configurations import Pythia14MConfig, load_safetensors


# Example usage
# Initialize the rafale_pythia model
rafale_pythia = DecoderWrapper(Pythia14MConfig)
rafale_pythia = load_safetensors(rafale_pythia, Pythia14MConfig)
ifdp = InferenceDatapipeline("EleutherAI/pythia-14m")
test_str = "Once upon a time,"

# Define input_ids (e.g., starting with a <bos> token)
# input_ids = torch.tensor([[Pythia14MConfig.bos_token_id]])  # Shape: (1, 1)

# Define maximum sequence length for generation
max_length = 32

# Generate sequence using greedy decoding
generated_sequence = greedy_decode(
    rafale_pythia,
    ifdp(test_str),
    max_length,
    Pythia14MConfig.eos_token_id,
    check_repeat_ngrams=True,
)

generated_str = ifdp.ids_to_str(generated_sequence["input_ids"])

print("Generated sequence:", generated_str)
