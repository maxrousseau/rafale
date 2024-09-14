import argparse
import yaml
import time

import torch
import torch.utils.data

import composer
from composer import Trainer
from composer.loggers import InMemoryLogger

from rafale.models.decoder import DecoderWrapper
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

parser = argparse.ArgumentParser(description="launch a training run")

parser.add_argument(
    "-r",
    "--run_config",
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
    with open(args.run_config, "r") as f:
        config = yaml.safe_load(f)

    run_name = config["run"]["name"]
    run_lr = config["run"]["lr"]
    run_n_epochs = config["run"]["n_epochs"]

    model_config_key = config["model"]["config"]
    model_type = config["model"]["type"]
    model_use_pretrained = config["model"]["use_pretrained"]

    data_pipeline_key = config["data"]["pipeline"]
    data_config_key = config["data"]["config"]

    # build & load the model
    model_config = model_config_dict[model_config_key]

    if model_type == "decoder":
        rafale_model = DecoderWrapper(model_config)
    elif model_type == "encoder":
        rafale_model = EncoderWrapper(model_config)
    else:
        raise TypeError(
            f"Model type {model_type} is not valid! Supports: encoder, decoder."
        )

    if model_use_pretrained:
        rafale_model = load_safetensors(rafale_model, model_config)

    # build & load the dataloader
    dataset_config = data_config_dict[data_config_key]
    data_pipeline = data_pipeline_dict[data_pipeline_key](**dataset_config)
    dataloaders = data_pipeline()

    # setup logging?
    logger_for_baseline = InMemoryLogger()

    # build trainer
    trainer = Trainer(
        model=rafale_model,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders["test"],
        optimizers=torch.optim.Adam(rafale_model.parameters(), lr=1e-5),
        max_duration=1,  # epochs
        device="cpu",
    )

    # launch
    trainer.fit()


if __name__ == "__main__":
    main()
