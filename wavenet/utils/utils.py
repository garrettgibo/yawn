"""Helper functions"""
import glob
import os

import torch
import wavenet.utils as utils
from pydub import AudioSegment

logger = utils.new_logger(__name__)


def get_data() -> None:
    """Download dataset into data directory"""
    pass


def convert_mp3_folder(input_folder: str, sample_rate: float) -> str:
    """Convert all mp3 files in folder into wav files"""
    logger.info(
        "Converting mp3 files located in [%s] to WAV with sample rate [%d]",
        input_folder,
        sample_rate,
    )

    output_folder = create_wav_folder(input_folder)
    mp3_files = glob.glob(input_folder + "/*.mp3")

    for mp3_path in mp3_files:
        mp3_file = mp3_path.split("/")[-1]
        mp3_to_wav(input_folder, output_folder, mp3_file, sample_rate)

    logger.info("Converted WAV files located in [%s]", output_folder)

    return output_folder


def mp3_to_wav(
    input_folder: str, output_folder: str, filename: str, sample_rate: float
) -> None:
    """Convert mp3 file to wav file"""
    mp3_full_path = os.path.join(input_folder, filename)
    wav_full_path = os.path.join(output_folder, filename.replace("mp3", "wav"))

    if not os.path.exists(wav_full_path):
        mp3 = AudioSegment.from_mp3(mp3_full_path)
        mp3.export(wav_full_path, format="wav", parameters=["-ar", f"{sample_rate}"])
        logger.debug("Converted %s to wav file", filename)
    else:
        logger.debug("%s already converted to wav file", filename)


def create_wav_folder(input_folder: str) -> str:
    """Create the wav output folder"""
    wav_folder = input_folder + "_wav"

    if not os.path.exists(wav_folder):
        os.makedirs(wav_folder)
        logger.info("Created wav folder: %s", wav_folder)

    return wav_folder


def load_model(model, optimizer, path: str):
    """Wrapper around loading a PyTorch model.

    Args:
        model: The model to be loaded into
        optimizer: The optimizer being used on the provided model
        path: Path to checkpoint to load

    """
    # Read checkpoint contents
    checkpoint = torch.load(path)

    # Load checkpoint contents into provded model and optimizer
    model.load_state_dict(checkpoint["net"])
    optimizer.load_state_dict(checkpoint["optim"])
    epoch = checkpoint["epoch"]

    logger.info("Restored checkpint from %s", path)

    return model, optimizer, epoch


def save_model(model, optimizer, epoch: int, name: str) -> None:
    """Wrapper around saving a PyTorch model.

    Args:
        model: The model that is saved.
        optimizer: The optimizer being used on the provided model
        epoch: Current epoch number
        name: Name of checkpoint

    """
    checkpoint = {
        "epoch": epoch,
        "net": model.state_dict(),
        "optim": optimizer.state_dict(),
    }
    torch.save(checkpoint, name)

    logger.info("Saving model at epoch [%d]", epoch)
