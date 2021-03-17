"""Helper functions"""
import glob
import os

import torch
import wavenet.utils as utils
from pydub import AudioSegment


def get_data() -> None:
    """Download dataset into data directory"""
    pass


class Mp3Converter:
    """Convert provided into WAV files"""

    def __init__(self, input_folder: str, sample_rate: int, log_level: int = 20):
        """Initialize an mp3 to WAV converter

        Args:
            input_folder: path to folder of mp3 files to convert
            sample_rate: desired sample rate to set WAV files to
            log_level: level to log at for converter. Default is 20 (logging.INFO)
        """
        self.logger = utils.new_logger("Mp3Converter", level=log_level)
        self.input_folder = input_folder
        self.sample_rate = sample_rate

    def convert(self) -> str:
        """Convert all mp3 files in folder into wav files"""
        self.logger.info(
            "Converting mp3 files located in [%s] to WAV with sample rate [%d]",
            self.input_folder,
            self.sample_rate,
        )

        output_folder = self.create_wav_folder()
        mp3_files = glob.glob(self.input_folder + "/*.mp3")

        for mp3_path in mp3_files:
            mp3_file = mp3_path.split("/")[-1]
            self.mp3_to_wav(output_folder, mp3_file)

        self.logger.info("Converted WAV files located in [%s]", output_folder)

        return output_folder

    def mp3_to_wav(self, output_folder: str, filename: str) -> None:
        """Convert mp3 file to wav file"""
        mp3_full_path = os.path.join(self.input_folder, filename)
        wav_full_path = os.path.join(output_folder, filename.replace("mp3", "wav"))

        if not os.path.exists(wav_full_path):
            mp3 = AudioSegment.from_mp3(mp3_full_path)
            mp3.export(
                wav_full_path, format="wav", parameters=["-ar", f"{self.sample_rate}"]
            )
            self.logger.debug("Converted %s to wav file", filename)
        else:
            self.logger.warning("%s already converted to wav file", filename)

    def create_wav_folder(self) -> str:
        """Create the wav output folder"""
        wav_folder = self.input_folder + "_wav"

        if not os.path.exists(wav_folder):
            os.makedirs(wav_folder)
            self.logger.info("Created wav folder: %s", wav_folder)
        else:
            self.logger.warning("Wav folder already created at %s", wav_folder)

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
