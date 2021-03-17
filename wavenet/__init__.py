"""Demo of WaveNet Pipeline"""
import wavenet.utils as utils
from torch.utils.data import DataLoader
from wavenet.modules import WaveNet
from wavenet.utils.data import WAVData


def demo(
    mp3_to_wav_cfg, dataset_cfg, dataloader_cfg, model_cfg, train_cfg, generator_cfg
):
    """Simple workflow for running through full data/train/evaluate pipeline"""
    # convert folder of mp3 files WAV files
    dataset_cfg["input_folder"] = utils.convert_mp3_folder(**mp3_to_wav_cfg)

    # Create dataset from WAV files
    dataloader_cfg["dataset"] = WAVData(**dataset_cfg)

    # create dataloader
    train_cfg["dataloader"] = DataLoader(**dataloader_cfg)

    # Create wavenet model
    train_cfg["model"] = WaveNet(**model_cfg)

    # Train model
    generator_cfg["model_path"] = utils.train(**train_cfg)

    # TODO: Generate music
    # generate_music(**generator_cfg)
