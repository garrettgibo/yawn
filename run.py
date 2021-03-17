#!/usr/bin/env python

import argparse
import json

import wavenet
import wavenet.utils as utils

logger = utils.new_logger("run")

TARGETS = {
    "train": wavenet.train_pipeline,
    "get-data": utils.get_data,
    "mp3-to-wav": utils.Mp3Converter,
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
