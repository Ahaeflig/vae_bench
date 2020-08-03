
import errno
import argparse
import os
import numpy as np
import tensorflow as tf
from runpy import run_path
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a model")
    parser.add_argument("--config_path", help=f'Path to configuration file', default='configs/default_config.py')
    args = parser.parse_args()
    config_path = args.config_path

    # Check that config file exit
    if not Path(config_path).is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_path)

    # Run code in config file to get config object
    settings = run_path(args.config_path)
    config = settings['config']

    print(config.keys())
