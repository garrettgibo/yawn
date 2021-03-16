#!/usr/bin/env python

import argparse
import json

from wavenet import demo
from wavenet.logger import new_logger
from wavenet.utils import get_data, convert_mp3_folder


logger = new_logger(__name__)

TARGETS = {
    "demo": demo,
    "get-data": get_data,
    "mp3-to-wav": convert_mp3_folder,
}

CONFIGS = {
    "demo": "config/demo.json",
    "get-data": ...,
    "mp3-to-wav": "config/mp3-to-wav.json"
}


def main():
    """Run WaveNet Pipeline/call designated targets"""
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=TARGETS.keys())
    args = parser.parse_args()

    with open(CONFIGS[args.target]) as config_file:
        config = json.load(config_file)

    # initiate target sequence with designated configuration
    TARGETS[args.target](**config)


if __name__ == "__main__":
    main()
