# 🧠 Deep MicroCompression

**Deep MicroCompression** is a research-driven project focused on optimizing deep neural network inference for **resource-constrained microcontrollers**. Inspired by Song Han’s *Deep Compression*, this project explores the combination of:

* **Structured pruning**
* **Low-bit quantization** (notably 4-bit)
* **Bit-level data packing**
* **Custom deployment kernels in C/C++**

The goal is to **reduce model size, memory access cost, and compute overhead** — all without significantly degrading accuracy — in environments where every byte and cycle counts.

---

## 💡 Core Ideas

* **Bit-Packed Quantization**:
  Instead of naively storing 4-bit weights in 8-bit containers, this project implements **bit-packing**, fitting two weights into a single byte — saving up to 50% space compared to standard 8-bit quantization.

* **Hardware-Conscious Design**:
  The deployment engine avoids expensive operations like floating-point division or branching, using **bitwise arithmetic and shift-based unpacking** for speed and predictability.

* **Layer-Level Customization**:
  Each layer (e.g., `Conv2d`, `Linear`, `ReLU`) is independently implemented in both PyTorch and C++, supporting quantization and deployment-specific logic.

* **C++ Runtime for Inference**:
  A minimal inference runtime written in C++ mirrors the Python-defined models and supports different quantization modes using conditional compilation.

---

## 📁 Project Structure Overview

```plaintext
development/         # PyTorch models and layer logic
  └── layers/        # Custom PyTorch layers (with quantization hooks)
  └── models/        # Model assembly (e.g., Sequential)
  └── utils.py       # Utilities for packing/processing

deployment/          # C++ inference engine
  └── layers/        # C++ versions of Conv, Linear, ReLU, etc.
  └── models/        # Model containers for deployment

tests/               # Various models (LeNet5, Sine, etc.) in C++ & Python
datasets/            # Input datasets (e.g., MNIST)
```

---

## 📌 Current Status

* 🛠️ Pruning, quantization and bit-packing implemented in key layers
* 🧪 Testing underway with models like LeNet5
* 📤 Conversion from PyTorch to C++ header is partially complete
* ❌ Full end-to-end automation (train → quantize → deploy) is **not yet integrated**

---

## 🎯 Objective

To demonstrate that **custom quantization-aware design + data layout optimization** can significantly boost the deployment efficiency of neural networks on **MCUs**, without relying on large ML runtimes or toolchains like TensorFlow Lite Micro.
