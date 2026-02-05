"""
Example: Simple benchmark for memory operations

This example benchmarks memory allocation, data transfer, and device operations.
"""

import time
import numpy as np
import paddle_iluvatar.device as device
import paddle_iluvatar.memory as memory
import paddle_iluvatar.stream as stream


def benchmark_memory_allocation(sizes, iterations=100):
    """Benchmark memory allocation performance"""
    print("\n=== Memory Allocation Benchmark ===")
    
    for size_mb in sizes:
        size_bytes = size_mb * 1024 * 1024
        
        start = time.time()
        for _ in range(iterations):
            mem = memory.malloc(size_bytes)
            del mem
        end = time.time()
        
        elapsed = (end - start) / iterations * 1000  # ms
        throughput = size_mb / (elapsed / 1000)  # MB/s
        
        print(f"Size: {size_mb:4d} MB | "
              f"Time: {elapsed:6.2f} ms | "
              f"Throughput: {throughput:8.2f} MB/s")


def benchmark_memory_transfer(sizes, iterations=10):
    """Benchmark host-device memory transfer"""
    print("\n=== Memory Transfer Benchmark ===")
    
    for size_mb in sizes:
        size_bytes = size_mb * 1024 * 1024
        size_floats = size_bytes // 4
        
        # Allocate host and device memory
        host_data = np.random.randn(size_floats).astype(np.float32)
        device_mem = memory.malloc(size_bytes)
        result = np.zeros_like(host_data)
        
        # Benchmark H2D
        start = time.time()
        for _ in range(iterations):
            memory.memcpy_h2d(device_mem, host_data)
        device.synchronize()
        end = time.time()
        
        h2d_time = (end - start) / iterations * 1000  # ms
        h2d_throughput = size_mb / (h2d_time / 1000)  # MB/s
        
        # Benchmark D2H
        start = time.time()
        for _ in range(iterations):
            memory.memcpy_d2h(result, device_mem)
        device.synchronize()
        end = time.time()
        
        d2h_time = (end - start) / iterations * 1000  # ms
        d2h_throughput = size_mb / (d2h_time / 1000)  # MB/s
        
        print(f"Size: {size_mb:4d} MB")
        print(f"  H2D: {h2d_time:6.2f} ms | {h2d_throughput:8.2f} MB/s")
        print(f"  D2H: {d2h_time:6.2f} ms | {d2h_throughput:8.2f} MB/s")
        
        del device_mem


def benchmark_stream_operations(iterations=1000):
    """Benchmark stream creation and synchronization"""
    print("\n=== Stream Operations Benchmark ===")
    
    # Benchmark stream creation
    start = time.time()
    streams = []
    for _ in range(iterations):
        s = stream.Stream()
        streams.append(s)
    end = time.time()
    
    creation_time = (end - start) / iterations * 1000000  # μs
    print(f"Stream creation: {creation_time:.2f} μs/stream")
    
    # Benchmark stream synchronization
    start = time.time()
    for s in streams:
        s.synchronize()
    end = time.time()
    
    sync_time = (end - start) / iterations * 1000000  # μs
    print(f"Stream sync: {sync_time:.2f} μs/stream")
    
    # Cleanup
    del streams
    
    # Benchmark event operations
    start = time.time()
    events = []
    for _ in range(iterations):
        e = stream.Event()
        events.append(e)
    end = time.time()
    
    event_creation_time = (end - start) / iterations * 1000000  # μs
    print(f"Event creation: {event_creation_time:.2f} μs/event")
    
    # Cleanup
    del events


def main():
    print("=" * 60)
    print("Paddle-Iluvatar Performance Benchmark")
    print("=" * 60)
    
    # Check GPU availability
    if not device.is_available():
        print("No Iluvatar GPU available!")
        return
    
    # Get device info
    props = device.get_device_properties(0)
    print(f"\nDevice: {props.name}")
    print(f"Memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"Compute Capability: {props.compute_capability_major}.{props.compute_capability_minor}")
    
    # Benchmark sizes
    allocation_sizes = [1, 10, 100, 500]  # MB
    transfer_sizes = [1, 10, 100]  # MB
    
    # Run benchmarks
    try:
        benchmark_memory_allocation(allocation_sizes)
        benchmark_memory_transfer(transfer_sizes)
        benchmark_stream_operations()
    except Exception as e:
        print(f"\nBenchmark error: {e}")
    
    # Get final memory stats
    free, total = memory.get_mem_info()
    print(f"\n=== Final Memory Status ===")
    print(f"Total: {total / (1024**3):.2f} GB")
    print(f"Free: {free / (1024**3):.2f} GB")
    print(f"Used: {(total - free) / (1024**3):.2f} GB")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
