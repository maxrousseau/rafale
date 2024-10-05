<div class="header" align="center">

# rafale

<div class="logo">
<p align="center">
<img src="./media/rafale-logo.png" alt="rafale-logo" width="200px" />
<br>
Rafale is a simple and opinionated transformer training CLI.
</p>
</div>

</div>


## Dependencies

Attempting to balance ergonomics and simplicity. This is meant to be easily hackable for research purposes.

```
torch, composer, datasets, tokenizers
```

## Purpose

Rafale provides an opinionated scaffolding for training transformers. It is solely built to be an efficient
learning/research tool. It is **not** a fully fledged library for large scale training.

It should be thought of as a starting point for research projects to bootstrap experiments on small LMs. The best way to
use rafale is to simply fork it and build on top of it for your specific purposes.

For large scale experiments other frameworks/libraries exist:
- torchtitan (Pytorch)
- torchtune (Pytorch)
- litGPT (LightningAI)
- GPT-NeoX (EleutherAI)
- nanotron (Huggingface)

## Usage

```sh

python rafale.main -r config.yaml

```

## Supported models


| Name    | Implemented | Inference test | Training test |
|:--------|:------------|:---------------|:--------------|
| BERT    | ✅          |                |               |
| RoBERTa | ✅          |                |               |
| Pythia  | ✅          | ✅             |               |
| minLSTM/minGRU      |             |                |               |


## Roadmap

v0.1 - MVP
- [x] single entrypoint CLI
- [ ] load weights from safetensors and include it in the config (BERT/RoBERTa and Pythia)
  - [x] pythia
  - [ ] BERT/RoBERTa (need to move from HF to safetensors)
- [x] Pythia KV-cache implementation
- [ ] greedy generation
- [ ] integration with lm-eval-harness
- [ ] clean up test suite*
- [ ] datapipes for CLM and MLM
  - local dataloader for now
  - [x] CLM tinystories
  - [ ] MLM tinystories
  - [ ] Imdb classification
- [-] ```tests``` for pythia and bert models on tinystories
- [x] ```main.py``` handles both training and evaluation (together or separately)
- [-] BERT/RoBERTa support (MLM objective)
- [x] Mosaic Composer/Trainer (see lightning-fabric simple trainer example and start from there)
  + bf16/fp16, gradient clipping, and gradient accumulation
  + building blocks are nn.Modules, specific models are ComposerModel classes with methods to load safetensor weights
    automatically (keep in a single separate file for each model)

+ [ ] move the testing in the notebook to a debug file in the modeling folder
+ optimizations : flash attn2, xformers layer_norm (triton) or RMSNorm, xformers fused_linear_layer
+ RMSNorm
+ **layerwise decay** for fine-tuning (https://kozodoi.me/blog/20220329/discriminative-lr)
+ *lm-eval-harness* integration guide:  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage



```
1) [x] tokenize
2) [x] concat and split w/ block size (pad w/ collator)
3) [x] save to disk {source}_{tokname}_bs{int}_len{int}
3) [x] data_collator: *next* pad (if desired), label shift right and return torch tensor # HF: does this in the model...
4) [ ] test with model training...


@TODO :: tiny stories but for MLM also...
```
