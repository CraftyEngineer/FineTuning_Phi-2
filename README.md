# Phi-2 LoRA Fine-tuning for Dialogue Summarization

This repository contains code for fine-tuning Microsoft's Phi-2 model (1.3B parameters) using LoRA (Low-Rank Adaptation) for dialogue summarization tasks, specifically on the DialogSum dataset.

## Features

- Efficient fine-tuning using 4-bit quantization (QLoRA)
- LoRA adaptation targeting query and value projection layers
- Training on a subset of the DialogSum dataset
- Example inference pipeline for text generation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Bitsandbytes
- Accelerate
- Datasets

## Usage

### Fine-tuning

The script performs the following steps:
1. Loads Phi-2 with 4-bit quantization
2. Prepares the model for k-bit training
3. Applies LoRA configuration
4. Loads and preprocesses the DialogSum dataset
5. Trains the model with specified parameters

## Training Configuration

- **Model**: microsoft/phi-2
- **Quantization**: 4-bit NF4 with double quantization
- **LoRA**:
  - Rank (r): 8
  - Alpha: 16
  - Target modules: q_proj, v_proj
  - Dropout: 0.05
- **Training**:
  - Batch size: 2 (per device)
  - Gradient accumulation: 4 steps
  - Learning rate: 2e-4
  - Max steps: 50
  - Warmup steps: 5

## Results

The model learns to generate concise summaries of dialogues in the "TL;DR" format after just 50 training steps.

Example input:
```
Dialogue:
Hi, how was your trip?
It was amazing! We visited 3 new countries.
```

Expected output:
```
TL;DR:
The person had an amazing trip visiting 3 new countries.
```

## Note

This implementation uses a small subset of the DialogSum dataset for demonstration purposes. For better results, consider training on the full dataset with more steps.

## License

The code is provided as-is under the MIT License. Note that the Phi-2 model has its own license terms from Microsoft.
