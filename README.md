# rafale

an opinionated encoder training library

## Dependencies

An attempt to keep them as minimal as possible (while also being ergonomic).

```
torch, lightning-fabric (or) accelerate, datasets, rich (eyecandy) ~~tokenizers will be removed~~
```

## Purpose

This package is solely built to be an efficient research tool. It will not support data preprocessing/handling
pipelines. It should be thought of as a starting point for research projects to bootstrap experiments on LMs.

Should be used pip installable and setup to be easily hackable to build on top of it.

Datasets should be preshuffled and pretokenized, only load it from disk and feed it to the dataloader with the collator function.

Mostly optimized for SLURM clusters.

```sh

rafale run -c config.yaml # set DEBUG=1 for a sanity check

```

## Roadmap

v0.1
- [ ] Local model weight loading
- [ ] Logging/Progress/Debug outputs with Rich library
- ~~RoBERTa BPE tokenizer with TikToken (compare w/ HF), on the fly tokenization to be handled by dataloader's
      collator (for MLM)~~
    - ~~model will be tied to the tokenizer, so dataloader will be defined after the model and use it's tokenizer~~
    - We don't want anything to do with preprocessing, all data should be split/augmented/shuffled/tokenized/etc. All we
      do with this tool is load it from disk, turn it to a tensor and send it to the model
- [ ] Local dataloader
- [ ] ```debug``` single batch debug
- [ ] ```main.py``` handles both training and evaluation (together or separately)
- [-] BERT/RoBERTa support (MLM objective)
  + [ ] move the testing in the notebook to a debug file in the modeling folder
  + **layerwise decay** for fine-tuning (https://kozodoi.me/blog/20220329/discriminative-lr)
  + optimizations : flash attn2, xformers layer_norm (triton) or RMSNorm, xformers fused_linear_layer
  + RMSNorm
- [ ] simple trainer (see lightning-fabric simple trainer example and start from there)
  + bf16/fp16, gradient clipping, and gradient accumulation

v0.2
- DeepSpeed ZeRO
  - Streaming dataloader
