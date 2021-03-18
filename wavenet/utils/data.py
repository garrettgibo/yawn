"""
data.py

Create a PyTorch dataset from a directory of wav files.
"""
import glob
from typing import List

import numpy as np
import wavenet.utils as utils
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset


class WAVData(Dataset):
    """Create PyTorch dataset from folder of WAV files"""

    def __init__(
        self,
        input_folder: str,
        input_length: int,
        output_length: int,
        mu: int = 255,
        num_classes: int = 256,
        log_level: int = 20,
    ):
        """Initialize the WAV dataset.

        Args:
            input_folder: path to folder of WAV files
            input_length: length to set for input to train on
            output_length: length to set for outputs
            num_classes: number classes to bin WAV input into
            log_level: logging level

        """
        self.logger = utils.new_logger(self.__class__.__name__, level=log_level)
        self.input_folder = input_folder
        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.mu = mu
        self.num_classes = num_classes
        self.track_list = glob.glob(input_folder + "/*.wav")
        self.data = self._create_data(self.track_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]["x"], self.data[idx]["y"]

    def _create_data(self, track_list: List[str]) -> np.ndarray:
        """Create dataset from list of track names"""
        self.logger.info("Creating dataset from files in [%s]", self.input_folder)
        data = []

        for track in track_list:
            extracted_data = self._extract_data(track)
            data.extend(extracted_data)
            self.logger.debug(
                "Added track [%d] points from [%s] to dataset",
                len(extracted_data),
                track.split("/")[-1],
            )

        self.logger.info(
            "Created dataset from files in [%s] with length: [%d]",
            self.input_folder,
            len(data),
        )

        return data

    def _extract_data(self, track_path: str):
        """Extract individual inputs and ouputs from a single track.

        The size of each data point is determined by self.input_length and
        self.output_length. This dictates how many individual samples to use as
        the input and how long the predicted output should be.

        """
        data = self._quantize(self._load_wav(track_path))
        data_length = self.input_length + self.output_length
        data_segments = []

        # Iterate over data as many times as allowed
        for i in range(0, len(data) - data_length, data_length):
            # Need data and target to have structure Tensor[num_channels, length]
            data_segments.append(
                {
                    "x": data[i : i + self.input_length].T,
                    "y": data[i + self.input_length : i + data_length].T,
                }
            )

        return data_segments

    def _load_wav(self, filename: str) -> np.ndarray:
        """Load a WAV file and return its contents as a numpy array"""
        sample_rate, data = wavfile.read(filename)
        data = np.array(data)

        # combine streams together
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        self.logger.debug(
            "Loaded track [%s] with sample rate [%d] and length [%d]",
            filename.split("/")[-1],
            sample_rate,
            len(data),
        )

        return data

    def _quantize(self, data: np.ndarray) -> np.ndarray:
        """Apply non-linear quantization to data."""
        data = self.mu_law_encoding(data)
        bins = np.linspace(-1, 1, self.num_classes)
        data_quantized = np.digitize(data, bins) - 1
        encoded = self._one_hot_encode(data_quantized)

        return encoded

    def mu_law_encoding(self, data: np.ndarray) -> np.ndarray:
        """Apply non-linear mu-law encoding"""
        data = np.sign(data) * np.log(1 + self.mu * np.abs(data)) / np.log(self.mu + 1)

        return data

    def mu_law_decoding(self, data: np.ndarray) -> np.ndarray:
        """Apply non-linear mu-law decoding"""
        data = (
            np.sign(data) * (np.exp(np.abs(data) * np.log(self.mu + 1)) - 1) / self.mu
        )

        return data

    def _one_hot_encode(self, data: np.ndarray) -> np.ndarray:
        """Create one hot encoding for WAV. """
        one_hot = np.zeros((data.size, self.num_classes), dtype=float)
        one_hot[np.arange(data.size), data.ravel()] = 1

        return one_hot


class WAVDataLoader(DataLoader):
    """Simple dataloader wrapper."""

    def __init__(
        self,
        dataset,
        shuffle: bool,
        batch_size: int,
        num_workers: int = 1,
        log_level: int = 20,
    ):
        super().__init__(
            dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
        )
        self.logger = utils.new_logger(self.__class__.__name__, level=log_level)
