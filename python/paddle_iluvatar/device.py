"""
Device management for Iluvatar GPUs

This module provides functions to interact with Iluvatar GPU devices.
"""

class DeviceProperties:
    """Properties of an Iluvatar GPU device"""
    def __init__(self):
        self.name = ""
        self.total_memory = 0
        self.compute_capability_major = 0
        self.compute_capability_minor = 0
        self.multi_processor_count = 0
        self.max_threads_per_block = 0
        self.max_threads_per_multi_processor = 0
    
    def __repr__(self):
        return (f"DeviceProperties(name='{self.name}', "
                f"total_memory={self.total_memory}, "
                f"compute_capability={self.compute_capability_major}."
                f"{self.compute_capability_minor})")


def get_device_count():
    """
    Get the number of available Iluvatar GPU devices.
    
    Returns:
        int: Number of devices
    """
    # Stub implementation
    # In actual implementation: return _C.get_device_count()
    return 1


def set_device(device_id):
    """
    Set the current Iluvatar GPU device.
    
    Args:
        device_id (int): Device ID to set as current
        
    Raises:
        ValueError: If device_id is invalid
    """
    if device_id < 0 or device_id >= get_device_count():
        raise ValueError(f"Invalid device ID: {device_id}")
    # Stub implementation
    # In actual implementation: _C.set_device(device_id)


def get_device():
    """
    Get the current Iluvatar GPU device ID.
    
    Returns:
        int: Current device ID
    """
    # Stub implementation
    # In actual implementation: return _C.get_device()
    return 0


def get_device_properties(device_id=None):
    """
    Get properties of an Iluvatar GPU device.
    
    Args:
        device_id (int, optional): Device ID. If None, uses current device.
        
    Returns:
        DeviceProperties: Device properties
    """
    if device_id is None:
        device_id = get_device()
    
    # Stub implementation
    props = DeviceProperties()
    props.name = "Iluvatar GPU"
    props.total_memory = 16 * 1024 * 1024 * 1024  # 16GB
    props.compute_capability_major = 1
    props.compute_capability_minor = 0
    props.multi_processor_count = 80
    props.max_threads_per_block = 1024
    props.max_threads_per_multi_processor = 2048
    
    return props


def synchronize():
    """
    Synchronize the current device.
    
    Waits for all operations on the current device to complete.
    """
    # Stub implementation
    # In actual implementation: _C.synchronize()
    pass


def is_available():
    """
    Check if Iluvatar GPU is available.
    
    Returns:
        bool: True if at least one device is available
    """
    try:
        return get_device_count() > 0
    except Exception:
        return False


__all__ = [
    'DeviceProperties',
    'get_device_count',
    'set_device',
    'get_device',
    'get_device_properties',
    'synchronize',
    'is_available',
]
