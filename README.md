# WaveNetClone

[![Lint Code Base](https://github.com/garrettgibo/WaveNetClone/actions/workflows/linter.yml/badge.svg)](https://github.com/garrettgibo/WaveNetClone/actions/workflows/linter.yml)

An implementation of the WaveNet (
[Blog](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio),
[Paper](https://arxiv.org/abs/1609.03499))
architecture in PyTorch that is trained on music from the [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset.

This implementation was inspired by the following sources:

- [Dankrushen/Wavenet-PyTorch](https://github.com/Dankrushen/Wavenet-PyTorch)
- [vincentherrmann/pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet)
- [ibab/tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)
- [musyoku/wavenet](https://github.com/musyoku/wavenet)

## Usage

For a full pipeline that goes through the following steps:

1. Convert directory of mp3 files to directory of WAV files at the specified
sample rate

2. Create PyTorch Dataset from directory of WAV files

3. Create PyTorch DataLoader from the custom dataset created in [2]

```sh
python run.py demo
```
