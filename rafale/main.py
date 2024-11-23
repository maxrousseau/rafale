from ast import Mod
import os
import warnings
import json
import hashlib
import argparse
import yaml
import time
from datetime import datetime

import torch
import torch.utils.data

import composer
from composer import Trainer, Time
from composer.loggers import InMemoryLogger, WandBLogger, FileLogger
from composer.algorithms import GradientClipping
from composer.callbacks import LRMonitor
from composer.optim.scheduler import (
    CosineAnnealingWithWarmupScheduler,
    CosineAnnealingScheduler,
    LinearScheduler,
    LinearWithWarmupScheduler,
)

from rafale.models.decoder import ComposerLM
from rafale.models.encoder import ComposerEncoderClassifier, ComposerMLM
from rafale.models.configurations import (
    load_safetensors,
    Pythia14MConfig,
    BertTinyConfig,
    BertConfig,
    RobertaConfig,
)

from rafale.caches import CHECKPOINT_CACHE_DIR, compute_config_hash, dump_config
from rafale.datapipe import TinyStoriesCausalNeoX, ImdbClsBERT, InferenceDatapipeline

ENV_VARS = {key: value for key, value in os.environ.items()}

parser = argparse.ArgumentParser(description="launch a training run")

parser.add_argument(
    "training_config",
    type=str,
    help="path to yaml run configuration file",
)
args = parser.parse_args()

model_config_dict = {
    "pythia14m": Pythia14MConfig,
    "berttiny": BertTinyConfig,
    "bert": BertConfig,
    "roberta": RobertaConfig,
}

data_pipeline_dict = {
    "tinystories_neox" : TinyStoriesCausalNeoX,
    "imdb_bert" : ImdbClsBERT
}

def save_output_run_status(status: int, output_dir: str):
    with open(os.path.join(output_dir, "run.out"), 'w') as f:
        f.write(str(status))


def check_run_status(output_dir: str) -> int:
    try:
        with open(os.path.join(output_dir, "run.out"), 'r') as f:
            content = f.read().strip()  # Read and strip whitespace
            if content in {'0', '1'}:  # Check if content is '0' or '1'
                return int(content)
            else:
                raise ValueError("Invalid content in run.out. Expected '0' or '1'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"run.out not found in directory: {output_dir}")

def main():
    # CONFIG ##################################################################
    with open(args.training_config, "r") as f:
        config = yaml.safe_load(f)

    run_name = config["run"]["name"]
    run_n_epochs = config["run"]["n_epochs"]
    run_seed = config["run"]["seed"]
    run_save_interval = config["run"]["save_interval"]
    run_eval_interval = config["run"]["eval_interval"]

    serialized_run_config = json.dumps(config)
    run_hash = compute_config_hash(serialized_run_config)

    run_clip_type = config["run"]["clip_type"]
    run_clip_value = float(config["run"]["clip_value"])

    device_bs = config["run"]["device_bs"]  # int or "auto"

    # schedule
    run_schedule_type = config["run"]["schedule"]
    run_max_lr = float(config["run"]["max_lr"])  # learning rate
    run_warmup_pct = float(config["run"]["warmup_pct"])

    run_eval_key = config["run"]["eval_key"]
    run_train_key = config["run"]["train_key"]

    if run_schedule_type == "cosine-warmup":
        run_scheduler = CosineAnnealingWithWarmupScheduler(
            t_warmup=Time(run_warmup_pct, "dur"), alpha_f=0.1
        )
    else:
        raise TypeError(
            f"Model type {model_type} is not valid! Supports: cosine-warmup.\nlinear, cosine, and linear-warmup planned"
        )

    model_config_key = config["run"]["model"]["config"]
    model_type = config["run"]["model"]["type"]
    model_use_pretrained = config["run"]["model"]["use_pretrained"]

    has_mode = False
    model_mode = None
    if "mode" in list(config["run"]["model"].keys()):
        has_mode = True
        model_mode = config["run"]["model"]["mode"] # specify mode for encoder

    has_n_classes = False
    model_n_classes = None
    if "n_classes" in list(config["run"]["model"].keys()):
        has_n_classes = True
        model_n_classes = config["run"]["model"]["n_classes"]

    data_pipeline_key = config["data"]["pipeline"]
    dataset_config = config["data"]["config"]

    # DATALOADERS #############################################################
    data_pipeline = data_pipeline_dict[data_pipeline_key](**dataset_config)
    dataloaders = data_pipeline()
    if "DATA" in ENV_VARS.keys() and ENV_VARS["DATA"] == "1":
        print("Data processing complete, exiting...")
        return 0  # just do data preprocessing if we pass DATA=1

    # MODEL #######################################################################
    model_config = model_config_dict[model_config_key]

    if model_type == "decoder":
        rafale_model = ComposerLM(model_config)
    elif model_type == "encoder":
        assert has_mode
        if model_mode == "cls":
            assert has_n_classes
            rafale_model = ComposerEncoderClassifier(model_config, mode=model_mode, num_classes=model_n_classes)
        else:
            raise NotImplementedError(f"Model mode: {model_mode} is not supported")
    else:
        raise TypeError(
            f"Model type {model_type} is not valid! Supports: encoder, decoder."
        )

    if model_use_pretrained:
        rafale_model.model = load_safetensors(rafale_model.model, model_config)

    # GRADIENT CLIPPING #######################################################
    clipping_type = "norm"  # can also be 'adaptive' or 'value'
    gradient_clip = GradientClipping(
        clipping_type=clipping_type, clipping_threshold=0.1
    )

    # DEVICES #################################################################
    device = "gpu" if torch.cuda.is_available() else "cpu"  # select the device
    if device == "gpu":
        run_precision = "amp_fp16"
    else:
        run_precision = "fp32"

    # DEBUG RUN ###############################################################
    if "DEBUG" in ENV_VARS.keys() and ENV_VARS["DEBUG"] == "1":
        from torch.utils.data import Subset, DataLoader, default_collate
        from datasets import Dataset

        # single batch, same for train and test 10 epochs
        bs = 4
        debug_batch = next(iter(dataloaders["train"]))
        debug_batch = Dataset.from_dict(
            {k: v[:bs] for k, v in debug_batch.items()}
        ).with_format("torch")
        debug_batch = DataLoader(
            debug_batch,
            batch_size=bs,
            shuffle=False,
            collate_fn=default_collate,
        )

        trainer = Trainer(
            model=rafale_model,
            seed=run_seed,
            train_dataloader=debug_batch,
            eval_dataloader=debug_batch,
            optimizers=torch.optim.AdamW(rafale_model.parameters(), lr=1e-4),
            max_duration=10,  # num epochs
            device=device,
            precision=run_precision,
            progress_bar=False,
            log_to_console=True,
            console_log_interval="1ep",
        )

        trainer.fit()

        return 0

    # LOGGING #################################################################
    # mem_logger = InMemoryLogger()
    # @TODO :: add some logging options in the yaml
    wandb_logger = WandBLogger(project="rafale", name=run_name)
    # file_logger = FileLogger(filename=f"{run_name}-{time}".txt)

    # TRAIN ###################################################################
    # training subset must have key "train" then whatever is called the validation subset (i.e. test, val, validation,
    # eval, etc) as long as there is only 1 other subset, we call it

    # check path*
    checkpoint_folder = os.path.abspath(
        os.path.join(CHECKPOINT_CACHE_DIR, f"{run_name}-{run_hash}/")
    )
    latest_checkpoint_load_path = None
    now = datetime.now()
    formatted_date = now.strftime(
        "%d" + "d" + "%m" + "m" + "%Y" + "y" + "_%H" + "h" + "%M" + "m"
    )  # Format it as DDdMMmYYYYy_HHhMMm

    if os.path.isdir(checkpoint_folder):
        warnings.warn(f"Run with same configuration already exists at location:\n\t{checkpoint_folder}")
        previously_failed = check_run_status(checkpoint_folder)

        if "FORCE" in ENV_VARS.keys() and ENV_VARS["FORCE"] == "1":
            # get datetime for checkpoint, directories are created by composer
            print("Forcing new training run from scratch!")

            checkpoint_folder = os.path.abspath(
                os.path.join(CHECKPOINT_CACHE_DIR, f"{run_name}-{run_hash}-{formatted_date}/")
            )
        elif previously_failed:
            # restart run from latest checkpoint*
            print(f"Previous run failed, attempting to restart from latest checkpoint!")
            latest_checkpoint_load_path = os.path.join(checkpoint_folder, "latest")

        else:
            print("ABORTING run launch! Use FORCE=1 to duplicate the training run with same configuration.")
            return 1

    trainer = Trainer(
        model=rafale_model,
        seed=run_seed,
        callbacks=[LRMonitor()],
        train_dataloader=dataloaders[run_train_key],
        eval_dataloader=dataloaders[run_eval_key],
        optimizers=torch.optim.AdamW(rafale_model.parameters(), lr=run_max_lr),
        max_duration=run_n_epochs,  # num epochs
        eval_interval=run_eval_interval,
        device_train_microbatch_size=device_bs,  # will handle gradient accumulation automatically
        device=device,
        loggers=[wandb_logger],
        precision=run_precision,
        progress_bar=True,
        schedulers=run_scheduler,
        algorithms=[gradient_clip],
        save_folder=checkpoint_folder,
        save_overwrite=True,
        save_latest_filename="latest",
        save_interval=run_save_interval,
        load_path=latest_checkpoint_load_path
    )

    dump_config(config, checkpoint_folder, name=f"run-{formatted_date}")

    # launch
    try:
        trainer.fit()
        save_output_run_status(0, checkpoint_folder)
        print(f"🍻 TRAINING COMPLETE\n💾 CHECKPOINTS SAVED AT LOCATION: {checkpoint_folder}")

    except:
        save_output_run_status(1, checkpoint_folder)
        print(f"❌ ERROR OCCURED, CHECKPOINTS/LOG AT {checkpoint_folder}")

if __name__ == "__main__":
    main()
