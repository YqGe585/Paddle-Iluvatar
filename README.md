# Paddle-Iluvatar

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)

IluvatarGPU适配Paddle - Iluvatar GPU adapter for PaddlePaddle deep learning framework.

天数智芯(Iluvatar)GPU 与飞桨(PaddlePaddle)深度学习框架的适配层。

## Overview | 概述

This project provides a GPU adapter that enables PaddlePaddle to leverage Iluvatar GPU hardware for deep learning workloads. The adapter implements core device management, memory operations, and stream management interfaces required for GPU-accelerated computing.

本项目提供了一个GPU适配器，使飞桨(PaddlePaddle)能够利用天数智芯(Iluvatar)GPU硬件进行深度学习计算。适配器实现了GPU加速计算所需的核心设备管理、内存操作和流管理接口。

## Features | 特性

- **Device Management | 设备管理**: Query and manage Iluvatar GPU devices
- **Memory Operations | 内存操作**: Allocate, free, and transfer data between host and device
- **Stream Management | 流管理**: Asynchronous operations with streams and events
- **Error Handling | 错误处理**: Comprehensive error checking and reporting
- **Python Interface | Python接口**: Easy-to-use Python API for GPU operations

## Architecture | 架构

```
paddle-iluvatar/
├── include/              # C++ header files
│   └── paddle_iluvatar/
│       ├── device.h      # Device management
│       ├── memory.h      # Memory operations
│       ├── stream.h      # Stream and event management
│       └── error.h       # Error handling
├── csrc/                 # C++ implementation
│   ├── device.cpp
│   ├── memory.cpp
│   ├── stream.cpp
│   └── error.cpp
├── python/               # Python bindings
│   └── paddle_iluvatar/
│       ├── device.py     # Device API
│       ├── memory.py     # Memory API
│       └── stream.py     # Stream API
├── tests/                # Unit tests
├── examples/             # Usage examples
└── docs/                 # Documentation
```

## Installation | 安装

### Prerequisites | 前置要求

- Python 3.6 or higher
- CMake 3.10 or higher
- C++ compiler with C++14 support
- Iluvatar GPU SDK (when available)

### Build from Source | 从源码构建

```bash
# Clone the repository
git clone https://github.com/YqGe585/Paddle-Iluvatar.git
cd Paddle-Iluvatar

# Install Python dependencies
pip install -r requirements.txt

# Build and install
python setup.py install
```

### For Development | 开发模式

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start | 快速开始

### Device Management | 设备管理

```python
import paddle_iluvatar.device as device

# Check if GPU is available
if device.is_available():
    print(f"Found {device.get_device_count()} Iluvatar GPU(s)")
    
    # Get device properties
    props = device.get_device_properties(0)
    print(f"Device: {props.name}")
    print(f"Memory: {props.total_memory / (1024**3):.2f} GB")
    
    # Set current device
    device.set_device(0)
    device.synchronize()
```

### Memory Operations | 内存操作

```python
import numpy as np
import paddle_iluvatar.memory as memory

# Allocate device memory
device_mem = memory.malloc(1024 * 1024)  # 1 MB

# Create host data
data = np.arange(1000, dtype=np.float32)

# Copy host to device
memory.memcpy_h2d(device_mem, data)

# Copy device to host
result = np.zeros_like(data)
memory.memcpy_d2h(result, device_mem)

# Get memory info
free, total = memory.get_mem_info()
print(f"Free: {free / (1024**3):.2f} GB / {total / (1024**3):.2f} GB")
```

### Stream Operations | 流操作

```python
import paddle_iluvatar.stream as stream

# Create stream
s = stream.Stream()

# Create events for timing
start = stream.Event()
end = stream.Event()

# Record events
start.record(s)
# ... GPU operations here ...
end.record(s)

# Synchronize and measure time
s.synchronize()
elapsed = start.elapsed_time(end)
print(f"Elapsed time: {elapsed:.3f} ms")
```

## Examples | 示例

The `examples/` directory contains several complete examples:

- `basic_device.py` - Device information and management
- `memory_operations.py` - Memory allocation and data transfer
- `stream_operations.py` - Asynchronous operations with streams

Run an example:

```bash
python examples/basic_device.py
```

## Testing | 测试

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_device.py

# Run with coverage
python -m pytest --cov=paddle_iluvatar tests/
```

## Documentation | 文档

For detailed documentation, please refer to:

- API Reference: See docstrings in Python modules
- Architecture Guide: [docs/architecture.md](docs/architecture.md)
- Integration Guide: [docs/integration.md](docs/integration.md)

## Current Status | 当前状态

**Note**: This is a stub implementation that provides the interface structure and API design. To enable actual Iluvatar GPU support, the following integration work is required:

**注意**: 这是一个存根实现，提供了接口结构和API设计。要启用实际的天数智芯GPU支持，需要完成以下集成工作：

1. **Iluvatar SDK Integration | SDK集成**: Replace stub implementations with actual Iluvatar GPU SDK calls
2. **Kernel Implementation | 算子实现**: Implement GPU kernels for common operations
3. **PaddlePaddle Integration | 飞桨集成**: Register as a custom device backend in PaddlePaddle
4. **Performance Optimization | 性能优化**: Optimize memory management and kernel execution

## Contributing | 贡献

Contributions are welcome! Please feel free to submit issues and pull requests.

欢迎贡献！请随时提交问题和拉取请求。

## License | 许可证

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## References | 参考

- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - PaddlePaddle Deep Learning Framework
- [Iluvatar CoreX](https://www.iluvatar.com/) - Iluvatar AI GPU Solutions

## Contact | 联系方式

For questions and support, please open an issue on GitHub.

如有问题和支持需求，请在GitHub上提交issue。
