# MEPO_PiP
Source Code for Multi-Domain Entropy-Weighted Policy Optimization (MEPO) and the Picture-in-Picture (PiP) Benchmark

## PiP Benchmark
The Picture-in-Picture (PiP) benchmark is a new benchmark for evaluating the multi-instances capabilities of MLLMs. 

## MEPO Algorithm

This approach encourages the model to learn a more balanced policy that performs well across all domains. The core logic can be found in `verl/trainer/core_algos.py`.

## Getting Started

**Setup Environment**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point for training is `verl/trainer/main.py`. You can run it with a configuration file.

Example of running a training script:
```bash
bash run.sh
```

The `examples` directory contains various yaml files for different models and algorithms.