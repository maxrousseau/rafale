# rafale

Rafale is (for now) a simple and opinionated transformer encoder training CLI.

## Dependencies

Attempting to balance ergonomics and simplicity. This is meant to be easily hackable for research purposes.

```
torch, lightning-fabric (or) accelerate, datasets, rich (eyecandy) ~~tokenizers will be removed~~
```

@TODO :: (check out this stream on HF accelerate)[https://www.youtube.com/watch?v=X-Jx5-YskKY]


## Purpose

This package is solely built to be an efficient research tool. It will not support data preprocessing/handling
pipelines. It should be thought of as a starting point for research projects to bootstrap experiments on small LMs.

Should be used pip installable via git and setup to be easily hackable to build on top of it.

Datasets should be preshuffled and pretokenized, only load it from disk and feed it to the dataloader with the collator
function.

## Usage

Mostly optimized for SLURM clusters.

```sh

rafale run -c config.yaml # set DEBUG=1 for a sanity check

```

## Supported models


| Name    | Implemented | Inference test | Training test |
|:--------|:------------|:---------------|:--------------|
| BERT    | [x]         | [ ]            | [ ]           |
| RoBERTa | [x]         | [ ]            | [ ]           |
| Pythia  | [ ]         | [ ]            | [ ]           |
| S4      | [ ]         | [ ]            | [ ]           |


## Roadmap

v0.1
- [ ] load weights from safetensors and include it in the config (BERT and Pythia)
- [ ] integration with lighteval (or LM eval harness)
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
- [ ] Mosaic Composer/Trainer (see lightning-fabric simple trainer example and start from there)
  + bf16/fp16, gradient clipping, and gradient accumulation
  + building blocks are nn.Modules, specific models are ComposerModel classes with methods to load safetensor weights
    automatically (keep in a single separate file for each model)

v0.2
- DeepSpeed ZeRO
  - Streaming dataloader
