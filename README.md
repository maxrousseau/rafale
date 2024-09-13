<div class="header" align="center">

# rafale

<div class="logo">
<p align="center">
<img src="./lil_logo/rafale-logo.png" alt="rafale-logo" width="200px" />
<br>
Rafale is (for now) a simple and opinionated transformer training CLI.
</p>
</div>

</div>


## Dependencies

Attempting to balance ergonomics and simplicity. This is meant to be easily hackable for research purposes.

```
torch, composer, datasets, tokenizers
```

*lm-eval-harness* integration guide:  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage


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
| BERT    | ✅          |                |               |
| RoBERTa | ✅          |                |               |
| Pythia  | ✅          | ✅             |               |
| S4      |             |                |               |


## Roadmap

v0.1 - MVP
- [ ] load weights from safetensors and include it in the config (BERT/RoBERTa and Pythia)
  - [x] pythia
  - [ ] BERT/RoBERTa
- [ ] Pythia KV-cache implementation
- [ ] integration with lm-eval-harness
- [ ] datapipes for CLM and MLM
  - local dataloader for now
- [ ] ```tests``` for pythia and bert models on tinystories
- [ ] ```main.py``` handles both training and evaluation (together or separately)
- [-] BERT/RoBERTa support (MLM objective)
- [ ] Mosaic Composer/Trainer (see lightning-fabric simple trainer example and start from there)
  + bf16/fp16, gradient clipping, and gradient accumulation
  + building blocks are nn.Modules, specific models are ComposerModel classes with methods to load safetensor weights
    automatically (keep in a single separate file for each model)

+ [ ] move the testing in the notebook to a debug file in the modeling folder
+ optimizations : flash attn2, xformers layer_norm (triton) or RMSNorm, xformers fused_linear_layer
+ RMSNorm
+ **layerwise decay** for fine-tuning (https://kozodoi.me/blog/20220329/discriminative-lr)
