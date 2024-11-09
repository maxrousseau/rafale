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
- **lingua** (Facebookresearch)
- torchtitan (Pytorch)
- torchtune (Pytorch)
- litGPT (LightningAI)
- GPT-NeoX (EleutherAI)
- nanotron (Huggingface)
- llm-foundry (MosaicML)

## Installation & Usage

Setup with ```uv``` ([install uv](https://github.com/astral-sh/uv)).
```sh
git clone <repo url>
cd rafale
uv venv
. .venv/bin/activate
uv pip install -r cuda-requirements.txt (or cpu-requirements.txt)
uv pip install -e .
```

Prepare a dataset

```sh
python rafale/main -r my_config.yaml
```

Launch a run with a configuration.

```sh
python rafale/main -r my_config.yaml
```

## Docs

Append this file ```rafale_docprompt.txt``` to your favorite LLM and ask away!

## Supported models


| Name        | Implemented | Inference test | Training test |
|:------------|:------------|:---------------|:--------------|
| BERT        | ✅          |                |               |
| RoBERTa     | ✅          |                |               |
| Pythia      | ✅          | ✅             | ✅           |
| CLIP/SigLIP | ⏳          |                |               |


## Roadmap

<details>
  <summary>v0.1</summary>


### v0.1 - initial release
- [x] single entrypoint CLI
- [ ] simple deploy/build
  - [x] CPU macos build - Ok, uv run works with this
  - [ ] SLURM compute-canada - TBD
  - [x] local linux machine - for now uv for venv + requirements.txt
    - NOTE: because uv still does not fully play well with pytorch recommend semi-manual setup*
- [ ] load weights from safetensors and include it in the config (BERT/RoBERTa and Pythia)
  - [x] pythia
  - [ ] BERT/RoBERTa (need to move from HF to safetensors)
    - [ ] MLM
    - [ ] Classification
- [x] Pythia KV-cache implementation
- [x] greedy generation
- [ ] clean up test suite
- [ ] datapipes for CLM and MLM
  - local dataloader for now
  - [x] CLM tinystories
  - [ ] MLM tinystories
  - [ ] Imdb classification
- [ ] ```tests``` for pythia and bert models on tinystories
- [x] ```main.py``` handles both training and evaluation (together or separately)
- [ ]  *lm-eval-harness* integration guide:  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage
- [ ] Mosaic Composer/Trainer (see lightning-fabric simple trainer example and start from there)
  + [x] fp16
  + [ ] gradient clipping
  + [x] gradient accumulation (automatically handled by composer)
  + [x] building blocks are nn.Modules, specific models are ComposerModel classes with methods to load safetensor weights
    automatically (keep in a single separate file for each model)
  + [x] set DEBUG=1 for 1 batch sanity check before launching a run

Datapipelines
1. [x] tokenize
2. [x] concat and split w/ block size (pad w/ collator)
3. [x] save to disk {source}_{tokname}_bs{int}_len{int}
4. [x] data_collator: *next* pad (if desired), label shift right and return torch tensor # HF: does this in the model...
5. [x] test with model training
6. [ ] tiny stories but for MLM also
</details>

<details>
  <summary>v1.0</summary>

### path to v1.0
cleanup and additional features
+ [ ] move the testing in the notebook to a debug file in the modeling folder
+ [ ] optimizations : flash attn2, xformers layer_norm (triton) or RMSNorm, xformers fused_linear_layer
+ [ ] try out schedulefree, SOAP, and other optimizers
+ [ ] **layerwise decay** for fine-tuning (https://kozodoi.me/blog/20220329/discriminative-lr)
+ [ ] multimodality CLIP
+ [ ] integration with lm-eval-harness

</details>
