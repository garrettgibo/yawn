"""Helper functions"""
import glob
import os

from pydub import AudioSegment

from .logger import logger


def get_data() -> None:
    """Download dataset into data directory"""
    pass


def convert_mp3_folder(input_folder: str) -> str:
    """Convert all mp3 files in folder into wav files"""
    output_folder = create_wav_folder(input_folder)

    mp3_files = glob.glob(input_folder + "/*.mp3")

    for mp3_path in mp3_files:
        mp3_file = mp3_path.split("/")[-1]
        mp3_to_wav(input_folder, output_folder, mp3_file)


def mp3_to_wav(input_folder: str, output_folder: str, filename: str) -> None:
    """Convert mp3 file to wav file"""
    mp3_full_path = os.path.join(input_folder, filename)
    wav_full_path = os.path.join(output_folder, filename.replace("mp3", "wav"))

    if not os.path.exists(wav_full_path):
        mp3 = AudioSegment.from_mp3(mp3_full_path)
        mp3.export(wav_full_path, format="wav")
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
