# Integration Guide

## Integrating with Iluvatar GPU SDK

This guide describes how to replace the stub implementation with actual Iluvatar GPU SDK calls.

## Prerequisites

1. Iluvatar GPU SDK installed
2. SDK headers and libraries accessible
3. Iluvatar GPU driver installed

## SDK Integration Steps

### 1. Update CMakeLists.txt

```cmake
# Find Iluvatar SDK
set(ILUVATAR_SDK_PATH $ENV{ILUVATAR_SDK_PATH})
if(NOT ILUVATAR_SDK_PATH)
    message(FATAL_ERROR "ILUVATAR_SDK_PATH not set")
endif()

# Include SDK headers
include_directories(${ILUVATAR_SDK_PATH}/include)

# Link SDK libraries
link_directories(${ILUVATAR_SDK_PATH}/lib)
target_link_libraries(paddle_iluvatar iluvatar_runtime)
```

### 2. Replace Device Management

In `csrc/device.cpp`, replace stub calls with SDK calls:

```cpp
#include <iluvatar_runtime.h>  // Actual SDK header

void Device::Initialize() {
    // Replace stub with actual SDK initialization
    iluInit();
    
    // Get device count from SDK
    iluGetDeviceCount(&g_device_count);
    
    g_initialized = true;
}

void Device::SetDevice(int device_id) {
    iluSetDevice(device_id);
    g_current_device = device_id;
}

void Device::Synchronize() {
    iluDeviceSynchronize();
}
```

### 3. Replace Memory Operations

In `csrc/memory.cpp`:

```cpp
void* Memory::Malloc(size_t size) {
    void* ptr;
    iluError_t error = iluMalloc(&ptr, size);
    if (error != iluSuccess) {
        throw Error(ERROR_OUT_OF_MEMORY, "iluMalloc failed");
    }
    return ptr;
}

void Memory::Free(void* ptr) {
    iluFree(ptr);
}

void Memory::MemcpyH2D(void* dst, const void* src, size_t size) {
    iluMemcpy(dst, src, size, iluMemcpyHostToDevice);
}
```

### 4. Replace Stream Operations

In `csrc/stream.cpp`:

```cpp
StreamHandle Stream::Create() {
    iluStream_t stream;
    iluStreamCreate(&stream);
    return reinterpret_cast<StreamHandle>(stream);
}

void Stream::Synchronize(StreamHandle stream) {
    iluStreamSynchronize(reinterpret_cast<iluStream_t>(stream));
}
```

### 5. Update Error Handling

Map SDK error codes to adapter error codes:

```cpp
ErrorCode MapSDKError(iluError_t sdk_error) {
    switch (sdk_error) {
        case iluSuccess: return SUCCESS;
        case iluErrorInvalidValue: return ERROR_INVALID_VALUE;
        case iluErrorOutOfMemory: return ERROR_OUT_OF_MEMORY;
        // ... map other errors
        default: return ERROR_UNKNOWN;
    }
}
```

## Implementing Operator Kernels

### Matrix Multiplication Example

Create `csrc/kernels/matmul.cu`:

```cpp
// Iluvatar GPU kernel for matrix multiplication
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C,
            int M, int N, int K, StreamHandle stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
```

## PaddlePaddle Custom Device Integration

### 1. Create Device Plugin

```cpp
// paddle_iluvatar_device.cpp
#include "paddle/fluid/platform/device/device_ext.h"
#include "paddle_iluvatar/device.h"

C_Status Init() {
    paddle_iluvatar::Device::Initialize();
    return C_SUCCESS;
}

C_Status GetDeviceCount(size_t* count) {
    *count = paddle_iluvatar::Device::GetDeviceCount();
    return C_SUCCESS;
}

// Implement other device callbacks...

void RegisterDevice() {
    CustomRuntimeParams params;
    params.size = sizeof(CustomRuntimeParams);
    params.device_type = "iluvatar";
    params.interface = {
        .init = Init,
        .get_device_count = GetDeviceCount,
        // ... other callbacks
    };
    RegisterCustomRuntime(params);
}
```

### 2. Register Operators

```python
# Register device with PaddlePaddle
import paddle
from paddle_iluvatar import device

# Set device
paddle.set_device('iluvatar:0')

# Use PaddlePaddle operators on Iluvatar GPU
x = paddle.randn([100, 100])
y = paddle.matmul(x, x.T)
```

## Testing with Real Hardware

### Unit Tests

```bash
# Run tests on actual hardware
ILUVATAR_DEVICE=0 python -m pytest tests/

# Run specific test
python tests/test_device.py
```

### Benchmarking

```bash
# Run performance benchmarks
python examples/benchmark_matmul.py --size 1024 --iterations 100
```

## Environment Variables

- `ILUVATAR_SDK_PATH`: Path to Iluvatar SDK
- `ILUVATAR_DEVICE`: Default device ID
- `ILUVATAR_VISIBLE_DEVICES`: Comma-separated list of visible devices

## Troubleshooting

### Device Not Found
- Check driver installation: `ilu-smi`
- Verify SDK path: `echo $ILUVATAR_SDK_PATH`
- Check device visibility: `echo $ILUVATAR_VISIBLE_DEVICES`

### Memory Errors
- Check available memory: `ilu-smi`
- Reduce batch size
- Enable memory pooling

### Performance Issues
- Use multiple streams for concurrency
- Enable kernel fusion
- Profile with Iluvatar profiler

## Next Steps

1. Implement remaining operator kernels
2. Add comprehensive tests
3. Optimize memory management
4. Add profiling support
5. Document performance characteristics
