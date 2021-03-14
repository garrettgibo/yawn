#!/usr/bin/env python

import argparse
import json

from wavenet.utils import get_data, convert_mp3_folder
from wavenet.logger import logger

TARGETS = {
    "get-data": get_data,
    "mp3-to-wav": convert_mp3_folder,
}

CONFIGS = {
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
    logger.info("Starting target sequence: %s", args.target)
    TARGETS[args.target](**config)

if __name__ == "__main__":
    main()
