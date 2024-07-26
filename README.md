# AMD AI PC WaveNet

This is an implementation of the WaveNet architecture on [AMD Ryzen™ AI powered PCs](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html).

## Directory Files
```
| *.ipynb
| *.py
| environment.yaml
| vaip_config.json
|- snapshots
|- models
|- train_samples
|- wav
```
- [train_demo.ipynb](./train_demo.ipynb): Demo for model training. Recommended to run on machines with GPU. Modified from repository [pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet).
- [deployment_demo.ipynb](./deployment_demo.ipynb): Demo for running the model on an AI PC.
- [generation_demo.ipynb](./generation_demo.ipynb): Demo for generation of audio on an AI PC.
- [WhisperWave.py](./WhisperWave.py): Run WhisperWave for testing.

## Runtime

The following runtime has been tested on **AMD Radeon 780M**.

|     **Session**    | **Inference (s)** | **Generation (sample/s)** |
|:------------------:|:-----------------:|:-------------------------:|
| PyTorch (CPU)      | 0.0954            | 0.0857                    |
| ONNX Runtime (CPU) | 0.0634            | 0.0649                    |
| ONNX Runtime (NPU) | 0.0601            | 0.0646                    |