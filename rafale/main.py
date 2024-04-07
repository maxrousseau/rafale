import argparse
import yaml


# generate datasetmodule/model given a ".py" file containing the setup code
from datasets.data_utils import DatasetWrapper
from models.model_utils import ModelWrapper

parser = argparse.ArgumentParser(description="launch a training run")
parser.add_argument(
    "-c", "--config", type=str, help="path to yaml configuration file", required=True
)
args = parser.parse_args()


def main():
    # load/parse yaml config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(config["run"]["name"])
    print(config["model"])

    # build & load the model
    # @HERE

    # build & load the dataloader

    # setup logging?

    # build trainer

    # launch

    return None


if __name__ == "__main__":
    main()
