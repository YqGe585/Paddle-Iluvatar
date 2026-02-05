# Architecture Guide

## Overview

The Paddle-Iluvatar adapter provides a bridge between the PaddlePaddle deep learning framework and Iluvatar GPU hardware. This document describes the architecture and design decisions.

## Component Architecture

### 1. C++ Core Layer

Located in `csrc/` and `include/paddle_iluvatar/`, this layer provides:

- **Device Management** (`device.cpp/h`): 
  - Device enumeration and initialization
  - Device context management
  - Device synchronization

- **Memory Management** (`memory.cpp/h`):
  - Device memory allocation/deallocation
  - Host-Device data transfer
  - Device-Device data transfer
  - Memory information queries

- **Stream Management** (`stream.cpp/h`):
  - Asynchronous stream operations
  - Event-based synchronization
  - Performance timing

- **Error Handling** (`error.cpp/h`):
  - Error code definitions
  - Exception handling
  - Error reporting

### 2. Python Binding Layer

Located in `python/paddle_iluvatar/`, this layer provides:

- High-level Python API for GPU operations
- NumPy integration for data transfer
- Pythonic error handling
- Resource management (RAII pattern)

### 3. Integration Points

The adapter is designed to integrate with PaddlePaddle through:

1. **Custom Device Registration**: Register Iluvatar as a device backend
2. **Operator Kernels**: GPU implementations of PaddlePaddle operators
3. **Memory Allocator**: Custom allocator for Iluvatar GPU memory
4. **Runtime Hooks**: Integration with PaddlePaddle's runtime

## Design Principles

### 1. Minimal Interface

The API surface is kept minimal to reduce maintenance burden:
- Core operations: device, memory, stream
- Clear separation of concerns
- Simple error handling

### 2. Stub Implementation

Current implementation is a stub that:
- Uses host memory to simulate device memory
- Provides correct API semantics
- Enables development and testing without hardware

### 3. SDK Independence

The design allows easy integration with Iluvatar SDK:
- All SDK calls are isolated in implementation files
- Headers define abstract interfaces
- Easy to swap stub with real implementation

### 4. Resource Safety

- RAII pattern for automatic cleanup
- No manual memory management in Python layer
- Exception-safe C++ code

## Memory Model

```
Host Memory           Device Memory
+-----------+         +-----------+
|  NumPy    |  H2D    |  Iluvatar |
|  Array    | ------> |   Memory  |
|           |         |           |
|           |  D2H    |           |
|           | <------ |           |
+-----------+         +-----------+
```

## Asynchronous Execution

```
Stream 1:  [Op A] ---> [Op B] ---> [Op C]
                           |
                        Event
                           |
Stream 2:  [Op D] ---> [Op E] <----+
```

- Multiple streams enable concurrent execution
- Events synchronize between streams
- Asynchronous transfers overlap with computation

## Future Enhancements

1. **Kernel Implementation**: GPU kernels for operators
2. **Memory Pool**: Efficient memory allocation
3. **Multi-GPU Support**: Multi-device coordination
4. **Profiling**: Performance analysis tools
5. **JIT Compilation**: Runtime kernel generation

## Integration Roadmap

### Phase 1: Core Infrastructure (Current)
- ✅ Device management API
- ✅ Memory operations API
- ✅ Stream management API
- ✅ Python bindings

### Phase 2: SDK Integration
- 🔲 Replace stub with Iluvatar SDK
- 🔲 Hardware device initialization
- 🔲 Real memory allocation
- 🔲 Stream and event implementation

### Phase 3: Operator Kernels
- 🔲 Matrix multiplication
- 🔲 Convolution
- 🔲 Activation functions
- 🔲 Pooling operations

### Phase 4: PaddlePaddle Integration
- 🔲 Custom device registration
- 🔲 Operator kernel registration
- 🔲 Memory allocator integration
- 🔲 End-to-end testing

### Phase 5: Optimization
- 🔲 Kernel fusion
- 🔲 Memory pool
- 🔲 Multi-stream execution
- 🔲 Performance profiling
