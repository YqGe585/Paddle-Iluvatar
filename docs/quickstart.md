# Quick Start Guide

This guide will help you get started with Paddle-Iluvatar quickly.

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Paddle-Iluvatar

```bash
# For users
pip install .

# For developers
pip install -e ".[dev]"
```

## Basic Usage

### Check GPU Availability

```python
import paddle_iluvatar.device as device

# Check if GPU is available
if device.is_available():
    print(f"Found {device.get_device_count()} GPU(s)")
else:
    print("No GPU found")
```

### Device Information

```python
# Get device properties
props = device.get_device_properties(0)
print(f"Device: {props.name}")
print(f"Memory: {props.total_memory / (1024**3):.2f} GB")
print(f"Compute Capability: {props.compute_capability_major}.{props.compute_capability_minor}")
```

### Memory Operations

```python
import numpy as np
import paddle_iluvatar.memory as memory

# Allocate device memory
device_mem = memory.malloc(1024 * 1024)  # 1 MB

# Create host data
data = np.arange(1000, dtype=np.float32)

# Transfer data
memory.memcpy_h2d(device_mem, data)

# Get result
result = np.zeros_like(data)
memory.memcpy_d2h(result, device_mem)
```

### Stream Operations

```python
import paddle_iluvatar.stream as stream

# Create stream
s = stream.Stream()

# Create events for timing
start = stream.Event()
end = stream.Event()

# Record operations
start.record(s)
# ... your GPU operations ...
end.record(s)

# Synchronize and get timing
s.synchronize()
elapsed = start.elapsed_time(end)
print(f"Elapsed time: {elapsed:.3f} ms")
```

## Running Examples

### Basic Device Example

```bash
python examples/basic_device.py
```

Expected output:
```
============================================================
Iluvatar GPU Device Information
============================================================

Number of Iluvatar GPUs: 1
Current device: 0

--- Device 0 ---
Name: Iluvatar GPU
Total Memory: 16.00 GB
...
```

### Memory Operations Example

```bash
python examples/memory_operations.py
```

### Stream Operations Example

```bash
python examples/stream_operations.py
```

### Benchmark

```bash
python examples/benchmark.py
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_device.py -v

# Run with coverage
python -m pytest --cov=paddle_iluvatar tests/
```

## Common Patterns

### Pattern 1: Simple Computation

```python
import numpy as np
import paddle_iluvatar.device as device
import paddle_iluvatar.memory as memory

# Initialize
device.set_device(0)

# Prepare data
x = np.random.randn(1000).astype(np.float32)
y = np.random.randn(1000).astype(np.float32)

# Allocate device memory
d_x = memory.malloc(x.nbytes)
d_y = memory.malloc(y.nbytes)
d_result = memory.malloc(x.nbytes)

# Transfer to device
memory.memcpy_h2d(d_x, x)
memory.memcpy_h2d(d_y, y)

# Perform computation (stub - would call GPU kernel)
# kernel_add(d_x, d_y, d_result)

# Get result
result = np.zeros_like(x)
memory.memcpy_d2h(result, d_result)

# Cleanup
device.synchronize()
```

### Pattern 2: Multi-Stream Execution

```python
import paddle_iluvatar.stream as stream

# Create multiple streams
stream1 = stream.Stream()
stream2 = stream.Stream()

# Queue operations on different streams
# operations on stream1 and stream2 can run concurrently

# Synchronize all streams
stream1.synchronize()
stream2.synchronize()
```

### Pattern 3: Performance Timing

```python
import paddle_iluvatar.stream as stream

s = stream.Stream()
start = stream.Event()
end = stream.Event()

# Time GPU operations
start.record(s)

# ... GPU operations ...

end.record(s)
s.synchronize()

elapsed_ms = start.elapsed_time(end)
print(f"Operation took {elapsed_ms:.3f} ms")
```

## Troubleshooting

### Issue: Import Error

```python
# Error: ModuleNotFoundError: No module named 'paddle_iluvatar'

# Solution: Install the package
pip install -e .
```

### Issue: Device Not Found

```python
# Error: No Iluvatar GPU available

# Solution: Check device availability
# 1. Verify driver installation
# 2. Check ILUVATAR_VISIBLE_DEVICES environment variable
# 3. Ensure SDK is properly installed
```

### Issue: Memory Error

```python
# Error: Out of memory

# Solution:
# 1. Check available memory
free, total = memory.get_mem_info()
print(f"Free: {free / (1024**3):.2f} GB")

# 2. Reduce allocation size
# 3. Free unused memory
del device_mem
```

## Next Steps

1. **Explore Examples**: Check out the `examples/` directory for more examples
2. **Read Documentation**: See `docs/` for detailed guides
3. **Run Tests**: Familiarize yourself with the test suite
4. **Contribute**: See `CONTRIBUTING.md` for contribution guidelines

## Getting Help

- Check the [API Reference](docs/api_reference.md)
- Read the [Architecture Guide](docs/architecture.md)
- See [Integration Guide](docs/integration.md) for SDK integration
- Open an issue on GitHub for bugs or questions

## Resources

- [PaddlePaddle Documentation](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
- [Iluvatar AI](https://www.iluvatar.com/)
- [GitHub Repository](https://github.com/YqGe585/Paddle-Iluvatar)
