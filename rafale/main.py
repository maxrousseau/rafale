import os
import argparse
import yaml
import time

import torch
import torch.utils.data

import composer
from composer import Trainer
from composer.loggers import InMemoryLogger, WandBLogger, FileLogger

from rafale.models.decoder import ComposerLM
from rafale.models.encoder import EncoderWrapper
from rafale.models.configurations import (
    load_safetensors,
    Pythia14MConfig,
    BertConfig,
    RobertaConfig,
)

from rafale.caches import CHECKPOINT_CACHE_DIR

from rafale.datapipe import TinyStoriesCausalNeoX
from rafale.data_configurations import MINI_TINYSTORIES, TINYSTORIES


ENVS_VARS = {key: value for key, value in os.environ.items()}

parser = argparse.ArgumentParser(description="launch a training run")

parser.add_argument(
    "-c",
    "--training_config",
    type=str,
    help="path to yaml run configuration file",
    required=True,
)
args = parser.parse_args()

model_config_dict = {
    "pythia14m": Pythia14MConfig,
    "bert": BertConfig,
    "roberta": RobertaConfig,
}

data_config_dict = {
    "mini_tinystories": MINI_TINYSTORIES,
    "tinystories": TINYSTORIES,
}

data_pipeline_dict = {
    "tinystories_neox": TinyStoriesCausalNeoX,
}


def main():
    # load/parse yaml config
    with open(args.training_config, "r") as f:
        config = yaml.safe_load(f)

    run_name = config["run"]["name"]
    run_lr = float(config["run"]["lr"])
    run_n_epochs = config["run"]["n_epochs"]
    run_seed = config["run"]["seed"]

    model_config_key = config["model"]["config"]
    model_type = config["model"]["type"]
    model_use_pretrained = config["model"]["use_pretrained"]

    data_pipeline_key = config["data"]["pipeline"]
    data_config_key = config["data"]["config"]

    # build & load the model
    model_config = model_config_dict[model_config_key]

    if model_type == "decoder":
        rafale_model = ComposerLM(model_config)
    elif model_type == "encoder":
        rafale_model = EncoderWrapper(model_config)
    else:
        raise TypeError(
            f"Model type {model_type} is not valid! Supports: encoder, decoder."
        )

    if model_use_pretrained:
        rafale_model.model = load_safetensors(rafale_model.model, model_config)

    # build & load the dataloader
    dataset_config = data_config_dict[data_config_key]
    data_pipeline = data_pipeline_dict[data_pipeline_key](**dataset_config)
    dataloaders = data_pipeline()

    # setup logging
    # mem_logger = InMemoryLogger()
    # @TODO :: add some logging options in the yaml
    wandb_logger = WandBLogger(project="rafale", name=run_name)
    # file_logger = FileLogger(filename=f"{run_name}-{time}".txt)

    # build trainer
    trainer = Trainer(
        model=rafale_model,
        seed=run_seed,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders["test"],
        optimizers=torch.optim.AdamW(rafale_model.parameters(), lr=run_lr),
        max_duration=run_n_epochs,  # num epochs
        eval_interval="50ba",  # default is 1ep !
        device="cpu",
        loggers=[wandb_logger],
        # precision="amp_fp16",
    )

    # @TODO :: implement model metric logging my modifying the class for pythia this will be perplexity (which is
    # provided by composer)...
    # https://docs.mosaicml.com/projects/composer/en/stable/composer_model.html
    # - [x] metrics
    # - [x] where are the logs ?
    # - [ ] DEBUG=1 for training runs
    # - [ ] lm-eval-harness integration

    # launch
    trainer.fit()


if __name__ == "__main__":
    main()
