def get_tokens_from_logits(logits, tokenizer=None):
    """
    return the prediced tokens for all of the inputs
    """
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Get the predicted token IDs
    predicted_token_ids = torch.argmax(probabilities, dim=-1)

    predicted_tokens = [
        tokenizer.convert_ids_to_tokens(seq.numpy())
        for seq in torch.unbind(predicted_token_ids, dim=0)
    ]
    return predicted_tokens
