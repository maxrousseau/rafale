import torch
import torch.nn.functional as F


def repeat_ngram(input_ids, ngram_max_length=4):
    """
    Clean up output by checking for repeated n-grams of length less than ngram_max_length.
    If it finds a repeated n-gram, it returns the n-gram; otherwise, it returns 0.

    Args:
        input_ids: Tensor of shape (seq_length,), the sequence of input token IDs.
        ngram_max_length: The maximum length of the n-gram to check for repetition.

    Returns:
        A list of tokens representing the repeated n-gram if found, otherwise 0.
    """
    ngram_found = False
    ngram_tokens = None

    # Create a set to store seen n-grams
    seen_ngrams = set()

    print(input_ids)
    # Iterate through possible n-gram lengths
    for n in range(1, ngram_max_length + 1):
        for i in range(len(input_ids) - n + 1):
            ngram = tuple(input_ids[i : i + n].tolist())
            if ngram in seen_ngrams:
                ngram_found = True
                ngram_tokens = list(ngram)
                break
            seen_ngrams.add(ngram)
        if ngram_found:
            break

    if ngram_found:
        return ngram_tokens
    else:
        return 0


def greedy_decode(model, batch, max_length, eos_token_id, check_repeat_ngrams=True):
    """
    Implements greedy decoding for the rafale transformer model.

    Args:
        model: The decoder model (e.g., DecoderWrapper).
        batch: Dictionary containing input_ids of shape (batch_size, seq_length), the input prompt.
        max_length: The maximum length of the generated sequence.
        eos_token_id: The ID of the end-of-sequence token.

    Returns:
        Dictionary containing input_ids of shape (batch_size, max_length) with the generated tokens.
    """
    batch_size = batch["input_ids"].size(0)
    if batch_size != 1:
        raise ValueError(
            "greedy_decode currently only supports batch_size=1. Provided batch_size: {}".format(
                batch_size
            )
        )

    input_seq_len = batch["input_ids"].size(1)
    kv_cache_list = None

    # Generate tokens until max_length or eos_token is generated
    for _ in range(max_length - input_seq_len):
        # Forward pass through the model
        outputs, kv_cache_list = model(batch, kv_cache_list)
        logits = outputs[:, -1, :]  # Get the logits for the last generated token

        # Greedily select the token with the highest probability
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)  # Shape: (1, 1)

        # Append the predicted token to the generated sequence
        batch["input_ids"] = torch.cat((batch["input_ids"], next_token), dim=1)

        # Check for repeated n-grams and stop if detected
        if check_repeat_ngrams:
            repeated_ngram = repeat_ngram(
                batch["input_ids"].squeeze(), ngram_max_length=4
            )
            if repeated_ngram != 0:
                print(repeated_ngram)
                break

        # Check if the sequence has generated the eos_token_id
        if next_token.item() == eos_token_id:
            break

    return batch
