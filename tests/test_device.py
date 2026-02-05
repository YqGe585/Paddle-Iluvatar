"""
Unit tests for device management
"""

import unittest
import paddle_iluvatar.device as device


class TestDevice(unittest.TestCase):
    
    def test_device_count(self):
        """Test getting device count"""
        count = device.get_device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)
    
    def test_is_available(self):
        """Test checking device availability"""
        available = device.is_available()
        self.assertIsInstance(available, bool)
    
    def test_get_set_device(self):
        """Test getting and setting current device"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        # Get current device
        current = device.get_device()
        self.assertIsInstance(current, int)
        self.assertGreaterEqual(current, 0)
        
        # Set device
        device.set_device(0)
        self.assertEqual(device.get_device(), 0)
    
    def test_invalid_device(self):
        """Test setting invalid device"""
        with self.assertRaises(ValueError):
            device.set_device(-1)
        
        with self.assertRaises(ValueError):
            device.set_device(999)
    
    def test_device_properties(self):
        """Test getting device properties"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        props = device.get_device_properties(0)
        self.assertIsInstance(props, device.DeviceProperties)
        self.assertIsInstance(props.name, str)
        self.assertGreater(props.total_memory, 0)
        self.assertGreaterEqual(props.compute_capability_major, 0)
        self.assertGreaterEqual(props.compute_capability_minor, 0)
    
    def test_synchronize(self):
        """Test device synchronization"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        # Should not raise an exception
        device.synchronize()


if __name__ == '__main__':
    unittest.main()
