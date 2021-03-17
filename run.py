#!/usr/bin/env python

import argparse
import json

from wavenet import train
from wavenet.utils import Mp3Converter, get_data, new_logger

logger = new_logger("run")

TARGETS = {
    "train": train,
    "get-data": get_data,
    "mp3-to-wav": Mp3Converter,
}

CONFIGS = {
    "train": "config/train.json",
    "get-data": ...,
    "mp3-to-wav": "config/mp3-to-wav.json",
}


def main():
    """Run WaveNet Pipeline/call designated targets"""
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=TARGETS.keys(), type=str, default="demo")
    parser.add_argument(
        "--debug", type=bool, default=False, help="Flag for showing debug messages"
    )

    args = parser.parse_args()

    with open(CONFIGS[args.target]) as config_file:
        config = json.load(config_file)
        if args.debug:
            config["log_level"] = 10
        else:
            config["log_level"] = 20

    # initiate target sequence with designated configuration
    TARGETS[args.target](**config)


if __name__ == "__main__":
    main()
