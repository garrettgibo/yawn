# YAWN [Work In Progress] &middot; [![Lint Code Base](https://github.com/garrettgibo/WaveNetClone/actions/workflows/linter.yml/badge.svg)](https://github.com/garrettgibo/WaveNetClone/actions/workflows/linter.yml)

Yet Another WaveNet (implemented in PyTorch)

This is an implementation of WaveNet that aims to be used for training on and
generating different forms of music.



## Usage

For a full pipeline that goes through the following steps:

1. Convert directory of mp3 files to directory of WAV files at the specified
sample rate

2. Create PyTorch Dataset from directory of WAV files

3. Create PyTorch DataLoader from the custom dataset created in [2]

4. Creates WaveNet Model

5. Trains model

```sh
python run.py demo
```

## References

Original WaveNet blog post and paper:

- [Blog](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
- [Paper](https://arxiv.org/abs/1609.03499)

This implementation is loosely inspired by the following sources:

- [Dankrushen/Wavenet-PyTorch](https://github.com/Dankrushen/Wavenet-PyTorch)
- [vincentherrmann/pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet)
- [ibab/tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)
- [musyoku/wavenet](https://github.com/musyoku/wavenet)
