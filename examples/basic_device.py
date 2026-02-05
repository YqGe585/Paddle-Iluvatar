"""
Example: Basic device operations

This example demonstrates basic device management operations
with Iluvatar GPU.
"""

import paddle_iluvatar.device as device


def main():
    print("=" * 60)
    print("Iluvatar GPU Device Information")
    print("=" * 60)
    
    # Check if GPU is available
    if not device.is_available():
        print("No Iluvatar GPU available!")
        return
    
    # Get device count
    count = device.get_device_count()
    print(f"\nNumber of Iluvatar GPUs: {count}")
    
    # Get current device
    current = device.get_device()
    print(f"Current device: {current}")
    
    # Get device properties for each device
    for i in range(count):
        print(f"\n--- Device {i} ---")
        props = device.get_device_properties(i)
        print(f"Name: {props.name}")
        print(f"Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"Compute Capability: {props.compute_capability_major}.{props.compute_capability_minor}")
        print(f"Multiprocessor Count: {props.multi_processor_count}")
        print(f"Max Threads per Block: {props.max_threads_per_block}")
        print(f"Max Threads per MP: {props.max_threads_per_multi_processor}")
    
    # Set device and synchronize
    print(f"\nSetting device to 0...")
    device.set_device(0)
    print(f"Synchronizing device...")
    device.synchronize()
    print("Done!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
