import torch
import torch.nn.functional as F


def repeat_ngram(input_ids, ngram_max_length=4):
    """
    clean up output on repeat of of ngram < max_length
    if it finds an ngram it return it otherwise it returns 0
    """
    ngram_found = False
    ngram_tokens = None

    if ngram_found:
        return ngram_tokens

    else:
        return 0


def greedy_decode(model, batch, max_length, eos_token_id):
    """
    n    Implements greedy decoding for the rafale transformer model.

        Args:
            model: The decoder model (e.g., DecoderWrapper).
            input_ids: Tensor of shape (batch_size, seq_length), the input prompt.
            max_length: The maximum length of the generated sequence.
            eos_token_id: The ID of the end-of-sequence token.

        Returns:
            Tensor of shape (batch_size, max_length) containing the generated tokens.
    """

    print(batch)
    batch_size = batch["input_ids"].size(0)
    input_seq_len = batch["input_ids"].size(1)
    kv_cache_list = None

    # Generate tokens until max_length or eos_token is generated for every batch
    for _ in range(max_length - input_seq_len):
        # Forward pass through the model
        outputs, kv_cache_list = model(batch, kv_cache_list)
        logits = outputs[:, -1, :]  # Get the logits for the last generated token

        # Greedily select the token with the highest probability
        next_token = torch.argmax(logits, dim=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)

        # Append the predicted token to the generated sequence
        batch["input_ids"] = torch.cat((batch["input_ids"], next_token), dim=1)

        # Check if all sequences have generated the eos_token_id
        if torch.all(next_token == eos_token_id):
            print("eos token")
            break

    return batch
