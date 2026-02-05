"""
Example: Memory operations

This example demonstrates memory allocation and data transfer
between host and Iluvatar GPU device.
"""

import numpy as np
import paddle_iluvatar.device as device
import paddle_iluvatar.memory as memory


def main():
    print("=" * 60)
    print("Iluvatar GPU Memory Operations")
    print("=" * 60)
    
    # Check if GPU is available
    if not device.is_available():
        print("No Iluvatar GPU available!")
        return
    
    # Get memory info
    free, total = memory.get_mem_info()
    print(f"\nDevice Memory:")
    print(f"  Total: {total / (1024**3):.2f} GB")
    print(f"  Free: {free / (1024**3):.2f} GB")
    print(f"  Used: {(total - free) / (1024**3):.2f} GB")
    
    # Allocate device memory
    print("\nAllocating 1 MB of device memory...")
    size = 1024 * 1024  # 1 MB
    device_mem = memory.malloc(size)
    print(f"Allocated {device_mem.size} bytes at {device_mem.ptr}")
    
    # Create host data
    print("\nCreating host data (array of 1000 floats)...")
    host_array = np.arange(1000, dtype=np.float32)
    print(f"Host array shape: {host_array.shape}")
    print(f"Host array size: {host_array.nbytes} bytes")
    print(f"First 10 elements: {host_array[:10]}")
    
    # Copy host to device
    print("\nCopying host data to device...")
    memory.memcpy_h2d(device_mem, host_array)
    print("Copy complete!")
    
    # Create destination array
    result_array = np.zeros_like(host_array)
    
    # Copy device to host
    print("\nCopying device data back to host...")
    memory.memcpy_d2h(result_array, device_mem)
    print("Copy complete!")
    print(f"First 10 elements: {result_array[:10]}")
    
    # Free memory
    print("\nFreeing device memory...")
    del device_mem
    print("Done!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
