import argparse
import json
import os
import warnings
from datetime import datetime

import torch
import torch.utils.data
import yaml
from composer import Time, Trainer
from composer.algorithms import GradientClipping
from composer.callbacks import LRMonitor
from composer.loggers import WandBLogger
from composer.optim.scheduler import (
    CosineAnnealingWithWarmupScheduler,
)

from rafale.caches import CHECKPOINT_CACHE_DIR, compute_config_hash, dump_config
from rafale.datapipe import ImdbClsBERT, TinyStoriesCausalNeoX
from rafale.models.configurations import (
    BertConfig,
    BertTinyConfig,
    Pythia14MConfig,
    RobertaConfig,
    load_safetensors,
)
from rafale.models.decoder import ComposerLM
from rafale.models.encoder import ComposerEncoderClassifier


def save_output_run_status(status: int, output_dir: str):
    with open(os.path.join(output_dir, "run.out"), "w") as f:
        f.write(str(status))

def check_run_status(output_dir: str) -> int:
    try:
        with open(os.path.join(output_dir, "run.out"), "r") as f:
            content = f.read().strip()  # Read and strip whitespace
            if content in {"0", "1"}:  # Check if content is '0' or '1'
                return int(content)
            else:
                raise ValueError("Invalid content in run.out. Expected '0' or '1'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"run.out not found in directory: {output_dir}")

class TrainingRun:
    def __init__(self):
        # CONFIG ##################################################################
        self.ENV_VARS = {key: value for key, value in os.environ.items()}

        self.model_config_dict = {
            "pythia14m": Pythia14MConfig,
            "berttiny": BertTinyConfig,
            "bert": BertConfig,
            "roberta": RobertaConfig,
        }

        self.data_pipeline_dict = {
            "tinystories_neox": TinyStoriesCausalNeoX,
            "imdb_bert": ImdbClsBERT,
        }

        parser = argparse.ArgumentParser(description="launch a training run")

        parser.add_argument(
            "training_config",
            type=str,
            help="path to yaml run configuration file",
        )
        args = parser.parse_args()
        with open(args.training_config, "r") as f:
            self.config = yaml.safe_load(f)

        self.run_name = self.config["run"]["name"]
        self.run_n_epochs = self.config["run"]["n_epochs"]
        self.run_seed = self.config["run"]["seed"]
        self.run_save_interval = self.config["run"]["save_interval"]
        self.run_eval_interval = self.config["run"]["eval_interval"]

        self.serialized_run_config = json.dumps(self.config)
        self.run_hash = compute_config_hash(self.serialized_run_config)

        self.run_clip_type = self.config["run"]["clip_type"]
        self.run_clip_value = float(self.config["run"]["clip_value"])

        self.device_bs = self.config["run"]["device_bs"]  # int or "auto"

        # schedule
        self.run_schedule_type = self.config["run"]["schedule"]
        self.run_max_lr = float(self.config["run"]["max_lr"])  # learning rate
        self.run_warmup_pct = float(self.config["run"]["warmup_pct"])

        self.run_eval_key = self.config["run"]["eval_key"]
        self.run_train_key = self.config["run"]["train_key"]

        self.model_config_key = self.config["run"]["model"]["config"]
        self.model_type = self.config["run"]["model"]["type"]
        self.model_use_pretrained = self.config["run"]["model"]["use_pretrained"]
        self.model_config = self.model_config_dict[self.model_config_key]


        if self.run_schedule_type == "cosine-warmup":
            self.run_scheduler = CosineAnnealingWithWarmupScheduler(
                t_warmup=Time(self.run_warmup_pct, "dur"), alpha_f=0.1
            )
        else:
            raise TypeError(
                f"Model type {self.model_type} is not valid! Supports: cosine-warmup.\nlinear, cosine, and linear-warmup planned"
            )

        self.has_mode = False
        self.model_mode = None
        if "mode" in list(self.config["run"]["model"].keys()):
            self.has_mode = True
            self.model_mode = self.config["run"]["model"][
                "mode"
            ]  # specify mode for encoder

        self.has_n_classes = False
        self.model_n_classes = None
        if "n_classes" in list(self.config["run"]["model"].keys()):
            self.has_n_classes = True
            self.model_n_classes = self.config["run"]["model"]["n_classes"]

        self.data_pipeline_key = self.config["data"]["pipeline"]
        self.dataset_config = self.config["data"]["config"]

    def _get_dataloaders(self):
        data_pipeline = self.data_pipeline_dict[self.data_pipeline_key](
            **self.dataset_config
        )
        dataloaders = data_pipeline()

        return dataloaders

    def _get_model(self):
        if self.model_type == "decoder":
            rafale_model = ComposerLM(self.model_config)
        elif self.model_type == "encoder":
            assert self.has_mode
            if self.model_mode == "cls":
                assert self.has_n_classes
                rafale_model = ComposerEncoderClassifier(
                    self.model_config, mode=self.model_mode, num_classes=self.model_n_classes
                )
            else:
                raise NotImplementedError(
                    f"Model mode: {self.model_mode} is not supported"
                )
        else:
            raise TypeError(
                f"Model type {self.model_type} is not valid! Supports: encoder, decoder."
            )

        if self.model_use_pretrained:
            rafale_model.model = load_safetensors(rafale_model.model, self.model_config)

        return rafale_model

    def _debug_run(self, dataloaders, model):
        from datasets import Dataset
        from torch.utils.data import DataLoader, default_collate

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
            model=model,
            seed=self.run_seed,
            train_dataloader=debug_batch,
            eval_dataloader=debug_batch,
            optimizers=torch.optim.AdamW(model.parameters(), lr=1e-4),
            max_duration=10,  # num epochs
            device=self.device,
            precision=self.run_precision,
            progress_bar=False,
            log_to_console=True,
            console_log_interval="1ep",
        )

        trainer.fit()

    def _get_trainer(
        self,
        dataloaders,
        model,
        loggers=[],
        checkpoint_folder=None,
        algo_list=[],
        latest_checkpoint_load_path=None,
    ):
        trainer = Trainer(
            model=model,
            seed=self.run_seed,
            callbacks=[LRMonitor()],
            train_dataloader=dataloaders[self.run_train_key],
            eval_dataloader=dataloaders[self.run_eval_key],
            optimizers=torch.optim.AdamW(model.parameters(), lr=self.run_max_lr),
            max_duration=self.run_n_epochs,  # num epochs
            eval_interval=self.run_eval_interval,
            device_train_microbatch_size=self.device_bs,  # will handle gradient accumulation automatically
            device=self.device,
            loggers=loggers,  # [wandb_logger],
            precision=self.run_precision,
            progress_bar=True,
            schedulers=self.run_scheduler,
            algorithms=algo_list,  # [gradient_clip],
            save_folder=checkpoint_folder,
            save_overwrite=True,
            save_latest_filename="latest",
            save_interval=self.run_save_interval,
            load_path=latest_checkpoint_load_path,
        )

        return trainer

    def __call__(self):
        # GRADIENT CLIPPING #######################################################
        clipping_type = "norm"  # can also be 'adaptive' or 'value'
        gradient_clip = GradientClipping(clipping_type=clipping_type, clipping_threshold=0.1)

        # DEVICES #################################################################
        self.device = "gpu" if torch.cuda.is_available() else "cpu"  # select the device
        if self.device == "gpu":
            self.run_precision = "amp_fp16"
        else:
            self.run_precision = "fp32"

        # DATALOADERS #############################################################
        dataloaders = self._get_dataloaders()
        if "DATA" in self.ENV_VARS.keys() and self.ENV_VARS["DATA"] == "1":
            print("Data processing complete, exiting...")
            return 0  # just do data preprocessing if we pass DATA=1

        # MODEL #######################################################################
        model = self._get_model()

        # DEBUG RUN ###############################################################
        if "DEBUG" in self.ENV_VARS.keys() and self.ENV_VARS["DEBUG"] == "1":
            self._debug_run(dataloaders, model)
            return 0

        # LOGGING #################################################################
        # mem_logger = InMemoryLogger()
        # @TODO :: add some logging options in the yaml
        wandb_logger = WandBLogger(project="rafale", name=self.run_name)
        # file_logger = FileLogger(filename=f"{run_name}-{time}".txt)

        # TRAIN ###################################################################
        # training subset must have key "train" then whatever is called the validation subset (i.e. test, val, validation,
        # eval, etc) as long as there is only 1 other subset, we call it

        # check path*
        latest_checkpoint_load_path = None
        checkpoint_folder = os.path.abspath(
            os.path.join(CHECKPOINT_CACHE_DIR, f"{self.run_name}-{self.run_hash}/")
        )
        now = datetime.now()
        formatted_date = now.strftime(
            "%d" + "d" + "%m" + "m" + "%Y" + "y" + "_%H" + "h" + "%M" + "m"
        )  # Format it as DDdMMmYYYYy_HHhMMm

        if os.path.isdir(checkpoint_folder):
            warnings.warn(
                f"Run with same configuration already exists at location:\n\t{checkpoint_folder}"
            )
            previously_failed = check_run_status(checkpoint_folder)

            if "FORCE" in self.ENV_VARS.keys() and self.ENV_VARS["FORCE"] == "1":
                # get datetime for checkpoint, directories are created by composer
                print("Forcing new training run from scratch!")

                checkpoint_folder = os.path.abspath(
                    os.path.join(
                        CHECKPOINT_CACHE_DIR,
                        f"{self.run_name}-{self.run_hash}-{formatted_date}/",
                    )
                )
            elif previously_failed:
                # restart run from latest checkpoint*
                print(
                    "Previous run failed, attempting to restart from latest checkpoint!"
                )
                os.path.join(checkpoint_folder, "latest")
                latest_checkpoint_load_path = os.path.join(checkpoint_folder, "latest")

            else:
                print(
                    "ABORTING run launch! Use FORCE=1 to duplicate the training run with same configuration."
                )
                return 1

        trainer = self._get_trainer(
            dataloaders,
            model,
            loggers=[wandb_logger],
            checkpoint_folder=None,
            algo_list=[gradient_clip],
            latest_checkpoint_load_path=latest_checkpoint_load_path)

        # launch
        try:
            trainer.fit()
            save_output_run_status(0, checkpoint_folder)
            print(
                f"🍻 TRAINING COMPLETE\n💾 CHECKPOINTS SAVED AT LOCATION: {checkpoint_folder}"
            )

        except:
            save_output_run_status(1, checkpoint_folder)
            print(f"❌ ERROR OCCURED, CHECKPOINTS/LOG AT {checkpoint_folder}")

        dump_config(self.config, checkpoint_folder, name=f"run-{formatted_date}")
