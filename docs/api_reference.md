# API Reference

## Python API

### paddle_iluvatar.device

Device management module for Iluvatar GPUs.

#### Functions

##### `get_device_count() -> int`

Get the number of available Iluvatar GPU devices.

**Returns:**
- `int`: Number of devices

**Example:**
```python
import paddle_iluvatar.device as device
count = device.get_device_count()
print(f"Found {count} GPU(s)")
```

##### `set_device(device_id: int) -> None`

Set the current Iluvatar GPU device.

**Parameters:**
- `device_id` (int): Device ID to set as current

**Raises:**
- `ValueError`: If device_id is invalid

**Example:**
```python
device.set_device(0)  # Use first GPU
```

##### `get_device() -> int`

Get the current Iluvatar GPU device ID.

**Returns:**
- `int`: Current device ID

**Example:**
```python
current = device.get_device()
```

##### `get_device_properties(device_id: int = None) -> DeviceProperties`

Get properties of an Iluvatar GPU device.

**Parameters:**
- `device_id` (int, optional): Device ID. If None, uses current device.

**Returns:**
- `DeviceProperties`: Device properties object

**Example:**
```python
props = device.get_device_properties(0)
print(f"Name: {props.name}")
print(f"Memory: {props.total_memory / (1024**3):.2f} GB")
```

##### `synchronize() -> None`

Synchronize the current device. Waits for all operations on the current device to complete.

**Example:**
```python
device.synchronize()
```

##### `is_available() -> bool`

Check if Iluvatar GPU is available.

**Returns:**
- `bool`: True if at least one device is available

**Example:**
```python
if device.is_available():
    print("GPU is available")
```

#### Classes

##### `DeviceProperties`

Properties of an Iluvatar GPU device.

**Attributes:**
- `name` (str): Device name
- `total_memory` (int): Total memory in bytes
- `compute_capability_major` (int): Major compute capability version
- `compute_capability_minor` (int): Minor compute capability version
- `multi_processor_count` (int): Number of multiprocessors
- `max_threads_per_block` (int): Maximum threads per block
- `max_threads_per_multi_processor` (int): Maximum threads per multiprocessor

---

### paddle_iluvatar.memory

Memory management module for Iluvatar GPUs.

#### Functions

##### `malloc(size: int) -> DeviceMemory`

Allocate device memory.

**Parameters:**
- `size` (int): Size in bytes to allocate

**Returns:**
- `DeviceMemory`: Allocated memory block

**Raises:**
- `RuntimeError`: If allocation fails

**Example:**
```python
import paddle_iluvatar.memory as memory
mem = memory.malloc(1024 * 1024)  # 1 MB
```

##### `memcpy_h2d(dst: DeviceMemory, src: np.ndarray, size: int = None) -> None`

Copy data from host to device.

**Parameters:**
- `dst`: Destination device memory
- `src`: Source host memory (numpy array or bytes)
- `size` (int, optional): Number of bytes to copy. If None, copies all.

**Example:**
```python
import numpy as np
data = np.arange(1000, dtype=np.float32)
device_mem = memory.malloc(data.nbytes)
memory.memcpy_h2d(device_mem, data)
```

##### `memcpy_d2h(dst: np.ndarray, src: DeviceMemory, size: int = None) -> None`

Copy data from device to host.

**Parameters:**
- `dst`: Destination host memory (numpy array or bytearray)
- `src`: Source device memory
- `size` (int, optional): Number of bytes to copy. If None, copies all.

**Example:**
```python
result = np.zeros(1000, dtype=np.float32)
memory.memcpy_d2h(result, device_mem)
```

##### `memcpy_d2d(dst: DeviceMemory, src: DeviceMemory, size: int = None) -> None`

Copy data from device to device.

**Parameters:**
- `dst`: Destination device memory
- `src`: Source device memory
- `size` (int, optional): Number of bytes to copy

**Example:**
```python
src_mem = memory.malloc(1024)
dst_mem = memory.malloc(1024)
memory.memcpy_d2d(dst_mem, src_mem)
```

##### `memset(ptr: DeviceMemory, value: int, size: int) -> None`

Set device memory to a value.

**Parameters:**
- `ptr`: Device memory pointer
- `value` (int): Value to set (0-255)
- `size` (int): Number of bytes to set

**Example:**
```python
mem = memory.malloc(1024)
memory.memset(mem, 0, 1024)
```

##### `get_mem_info() -> Tuple[int, int]`

Get memory information for the current device.

**Returns:**
- `tuple`: (free_memory, total_memory) in bytes

**Example:**
```python
free, total = memory.get_mem_info()
print(f"Free: {free / (1024**3):.2f} GB")
print(f"Total: {total / (1024**3):.2f} GB")
```

#### Classes

##### `DeviceMemory`

Represents a block of device memory.

**Attributes:**
- `size` (int): Memory block size in bytes
- `ptr`: Memory pointer

**Methods:**
- `__init__(size: int)`: Allocate device memory
- `__del__()`: Free device memory automatically

---

### paddle_iluvatar.stream

Stream management module for asynchronous operations.

#### Classes

##### `Stream`

Represents a GPU stream for asynchronous operations.

**Methods:**

###### `__init__()`

Create a new stream.

**Example:**
```python
import paddle_iluvatar.stream as stream
s = stream.Stream()
```

###### `synchronize() -> None`

Wait for all operations in this stream to complete.

**Example:**
```python
s.synchronize()
```

###### `query() -> bool`

Check if all operations in this stream have completed.

**Returns:**
- `bool`: True if completed, False otherwise

**Example:**
```python
if s.query():
    print("Stream completed")
```

###### `wait() -> None`

Wait for this stream to complete (alias for synchronize).

**Attributes:**
- `handle`: Get the stream handle

##### `Event`

Represents a GPU event for timing and synchronization.

**Methods:**

###### `__init__()`

Create a new event.

**Example:**
```python
e = stream.Event()
```

###### `record(stream: Stream = None) -> None`

Record the event on a stream.

**Parameters:**
- `stream` (Stream, optional): Stream to record on. If None, uses default.

**Example:**
```python
e.record(s)
```

###### `synchronize() -> None`

Wait for the event to complete.

**Example:**
```python
e.synchronize()
```

###### `query() -> bool`

Check if the event has completed.

**Returns:**
- `bool`: True if completed, False otherwise

###### `elapsed_time(end_event: Event) -> float`

Measure elapsed time between this event and another.

**Parameters:**
- `end_event` (Event): End event

**Returns:**
- `float`: Elapsed time in milliseconds

**Example:**
```python
start = stream.Event()
end = stream.Event()
start.record(s)
# ... GPU operations ...
end.record(s)
s.synchronize()
elapsed = start.elapsed_time(end)
print(f"Elapsed: {elapsed:.3f} ms")
```

**Attributes:**
- `handle`: Get the event handle

---

## C++ API

### paddle_iluvatar::Device

Device management class.

#### Static Methods

- `static int GetDeviceCount()`: Get number of available devices
- `static void SetDevice(int device_id)`: Set current device
- `static int GetDevice()`: Get current device ID
- `static void GetDeviceProperties(DeviceProperties* props, int device_id)`: Get device properties
- `static void Synchronize()`: Synchronize device
- `static void Initialize()`: Initialize device
- `static void Finalize()`: Finalize device

### paddle_iluvatar::Memory

Memory management class.

#### Static Methods

- `static void* Malloc(size_t size)`: Allocate device memory
- `static void Free(void* ptr)`: Free device memory
- `static void MemcpyH2D(void* dst, const void* src, size_t size)`: Copy host to device
- `static void MemcpyD2H(void* dst, const void* src, size_t size)`: Copy device to host
- `static void MemcpyD2D(void* dst, const void* src, size_t size)`: Copy device to device
- `static void Memset(void* ptr, int value, size_t size)`: Set device memory
- `static void GetMemInfo(size_t* free, size_t* total)`: Get memory info

### paddle_iluvatar::Stream

Stream management class.

#### Static Methods

- `static StreamHandle Create()`: Create a stream
- `static void Destroy(StreamHandle stream)`: Destroy a stream
- `static void Synchronize(StreamHandle stream)`: Synchronize a stream
- `static bool Query(StreamHandle stream)`: Query stream status
- `static void Wait(StreamHandle stream)`: Wait for stream completion

### paddle_iluvatar::Event

Event management class.

#### Static Methods

- `static EventHandle Create()`: Create an event
- `static void Destroy(EventHandle event)`: Destroy an event
- `static void Record(EventHandle event, StreamHandle stream)`: Record event on stream
- `static void Synchronize(EventHandle event)`: Synchronize event
- `static bool Query(EventHandle event)`: Query event status
- `static float ElapsedTime(EventHandle start, EventHandle end)`: Measure elapsed time

### Error Handling

#### Error Codes

```cpp
enum ErrorCode {
    SUCCESS = 0,
    ERROR_INVALID_VALUE = 1,
    ERROR_OUT_OF_MEMORY = 2,
    ERROR_NOT_INITIALIZED = 3,
    ERROR_DEVICE_NOT_FOUND = 4,
    ERROR_INVALID_DEVICE = 5,
    ERROR_MEMORY_COPY_FAILED = 6,
    ERROR_LAUNCH_FAILED = 7,
    ERROR_SYNCHRONIZATION_FAILED = 8,
    ERROR_UNKNOWN = 999
};
```

#### Macro

- `PADDLE_ILUVATAR_CHECK(call)`: Check error and throw on failure

**Example:**
```cpp
#include "paddle_iluvatar/device.h"
#include "paddle_iluvatar/memory.h"

void example() {
    // Initialize device
    paddle_iluvatar::Device::Initialize();
    
    // Allocate memory
    size_t size = 1024 * 1024;
    void* ptr = paddle_iluvatar::Memory::Malloc(size);
    
    // Use memory...
    
    // Free memory
    paddle_iluvatar::Memory::Free(ptr);
    
    // Finalize
    paddle_iluvatar::Device::Finalize();
}
```
