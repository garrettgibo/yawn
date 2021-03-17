"""Demo of WaveNet Pipeline"""
import wavenet.utils as utils
from wavenet.modules import WaveNet
from wavenet.utils.data import WAVData, WAVDataLoader


def train(
    mp3_to_wav_cfg: dict,
    dataset_cfg: dict,
    dataloader_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    generator_cfg: dict,
    log_level: int,
):
    """Simple workflow for running through full data/train/evaluate pipeline"""
    mp3_to_wav_cfg["log_level"] = log_level
    dataset_cfg["log_level"] = log_level
    dataloader_cfg["log_level"] = log_level
    model_cfg["log_level"] = log_level
    train_cfg["log_level"] = log_level
    generator_cfg["log_level"] = log_level

    # convert folder of mp3 files WAV files
    dataset_cfg["input_folder"] = utils.Mp3Converter(**mp3_to_wav_cfg).convert()

    # Create dataset from WAV files
    dataloader_cfg["dataset"] = WAVData(**dataset_cfg)

    # create dataloader
    train_cfg["train_dataloader"] = WAVDataLoader(**dataloader_cfg)

    # Create wavenet model
    train_cfg["model"] = WaveNet(**model_cfg)

    # Train model
    generator_cfg["model_path"] = utils.train(**train_cfg)

    # TODO: Generate music
    # generate_music(**generator_cfg)
