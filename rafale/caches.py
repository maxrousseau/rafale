import os
import json
import hashlib

DATA_CACHE_DIR = os.path.expanduser("~/.rafale_cache/data/")
MODEL_CACHE_DIR = os.path.expanduser("~/.rafale_cache/models/")
CHECKPOINT_CACHE_DIR = os.path.expanduser("~/.rafale_cache/checkpoints/")

def compute_config_hash(serialized_config):
    return hashlib.sha256(serialized_config.encode('utf-8')).hexdigest()


def dump_config(config_dict, output_dir, name):
    out_file = os.path.join(output_dir, f"{name}_config.json")
    with open(out_file, 'w') as f:
        json.dump(config_dict, f, indent=4)
