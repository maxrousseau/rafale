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

## 💡Purpose

Rafale provides an opinionated scaffolding for training transformers. It is solely built to be an efficient
learning/research tool. It is **not** a fully fledged library for large scale training.

It should be thought of as a starting point for research projects to bootstrap experiments on small LMs. The best way to
use rafale is to simply fork it and build on top of it for your specific purposes.

### Core dependencies

Attempting to balance ergonomics and simplicity. This is meant to be easily hackable for research purposes.

```
torch, composer, datasets, tokenizers
```

## 🚀 Installation & Usage

Setup with ```uv``` ([install uv](https://github.com/astral-sh/uv)).
```sh
$ git clone <repo url>
$ cd rafale
$ uv venv
$ . .venv/bin/activate
$ uv pip install -r cuda-requirements.txt (or cpu-requirements.txt)
$ uv pip install -e .
```

Launch a run with a configuration.

```sh
$ python rafale/main -c test/pythia_tinystories.yaml
```

What if I just want to prepare my dataset? ```DATA=1``` will run the data preparation and caching pipeline without
launching the training run.

```sh
$ DATA=1 python rafale/main -c test/pythia_tinystories.yaml
```

What if I want to test my model to make sure that its learning? ```DEBUG=1``` will run 10 epochs on a single training
batch (same for train/eval), the model should fit quickly if there are no bugs in the implementation.

```sh
$ DEBUG=1 python rafale/main -c test/pythia_tinystories.yaml
```


### 🔧 Under the hood

The goal of rafale is to provide a single entry point for data preparation and training. You configure the model and
dataset. Then call the training job.

When calling a run, first we run the datapipepline. If the dataset has already been processed (tokenized, padded,
chunked, etc.), it will be loaded from the cache (default location is ```~/.rafale_cache```.

> [!NOTE]
> #### Adding a new model
> To add a new model, you need write a new configuration to ```rafale/models/configurations.py```, and add it's key to
> ```model_config_dict``` in ```rafale/main.py```.
>
> Look at the ```ComposerLM``` wrapper class in ```rafale/models/decoder.py``` to check if all your building blocks are
> there. Otherwise you may need to modify/write a new wrapper.
>
> #### Adding a new datapipeline
>
> If the dataset is hosted on huggingface, simply use git lfs to clone the repo locally or use the repo name as the
> dataset path. Same goes for tokenizers since we use their tokenizer implementation.
>
> You will need to add a new datapipeline class in ```rafale/datapipes.py``` where the ```_prepare``` method all data
> preprocessing (tokenization, chunking, truncation, etc.) **EXCEPT** padding. Padding will be performed by the datacollator.

### 📕 Docs

Append this file ```llm-docprompt.txt``` to your favorite LLM and ask away.

### 🦾 Supported models


| Name        | Implemented | Inference test | Training test |
|:------------|:------------|:---------------|:--------------|
| BERT        | ✅          |                |               |
| RoBERTa     | ✅          |                |               |
| Pythia      | ✅          | ✅             | ✅           |
| CLIP/SigLIP | ⏳          |                |               |


## 🔮 Roadmap

<details>
  <summary>v0.1</summary>


### v0.1 - initial release
- [x] single entrypoint CLI
- [ ] simple deploy/build
  - [x] CPU macos build - Ok, uv run works with this
  - [x] local linux machine - for now uv for venv + requirements.txt
  - [ ] SLURM compute-canada - TBD
    - NOTE: because uv still does not fully play well with pytorch recommend semi-manual setup*
- [ ] load weights from safetensors and include it in the config (BERT/RoBERTa and Pythia)
  - [x] pythia
  - [ ] BERT/RoBERTa (need to move from HF to safetensors)
    - [ ] MLM
    - [ ] Classification
- [x] Pythia KV-cache implementation
- [x] greedy generation
- [ ] datapipes for CLM and MLM
  - local dataloader for now
  - [x] CLM tinystories
  - [ ] MLM tinystories
  - [ ] Imdb classification
- [x] ```main.py``` handles both training and evaluation (together or separately)
- [x] Mosaic Composer/Trainer
  + [x] fp16
  + [x] gradient clipping
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
- [ ] clean up ```tests``` for pythia and bert models on tinystories
- [ ] move the testing in the notebook to a debug file in the modeling folder
- [ ] optimizations : flash attn2, xformers layer_norm (triton) or RMSNorm, xformers fused_linear_layer
- [ ] try out schedulefree, SOAP, and other optimizers
- [ ] **layerwise decay** for fine-tuning (https://kozodoi.me/blog/20220329/discriminative-lr)
- [ ] multimodality CLIP
- [ ] integration with lm-eval-harness (guide)[https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage]

</details>

## I am GPU-rich what do I use?

For large scale experiments other frameworks/libraries exist:
- lingua (Facebookresearch)
- torchtitan (Pytorch)
- torchtune (Pytorch)
- litGPT (LightningAI)
- GPT-NeoX (EleutherAI)
- nanotron (Huggingface)
- llm-foundry (MosaicML)
