# AMD AI PC WaveNet

This is an implementation of the WaveNet architecture on [AMD Ryzen™ AI powered PCs](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html).

## Files
- *.ipynb: Jupyter notebook for demos.
- *.py: Python scripts for WaveNet. Modified from repository [pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet).
- environment.yaml: Conda environment.
- vaip_config.json: Configuration json file for deployment on NPU.
- snapshots: Trained PyTorch models.
- models: ONNX models.
- train_samples: Training data.

## Demo
- [train_demo.ipynb](./train_demo.ipynb): Demo for model training. Recommended to run on machines with GPU. Modified from repository [pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet).
- [deployment_demo.ipynb](./deployment_demo.ipynb): Demo for running the model on an AI PC.
- [generation_demo.ipynb](./generation_demo.ipynb): Demo for generation of audio on an AI PC.

## Runtime

The following runtime has been tested on **AMD Radeon 780M**.

|     **Session**    | **Inference (s)** | **Generation (s)** |
|:------------------:|:-----------------:|:------------------:|
| PyTorch (CPU)      | 4.60e-03          | 829.6              |
| ONNX Runtime (CPU) | 4.80e-06          | 24.9               |
| ONNX Runtime (NPU) | 6.37e-06          | 25.8               |