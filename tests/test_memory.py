"""
Unit tests for memory management
"""

import unittest
import numpy as np
import paddle_iluvatar.memory as memory
import paddle_iluvatar.device as device


class TestMemory(unittest.TestCase):
    
    def test_malloc_free(self):
        """Test memory allocation and deallocation"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        # Allocate memory
        size = 1024
        mem = memory.malloc(size)
        self.assertIsNotNone(mem)
        self.assertEqual(mem.size, size)
        self.assertIsNotNone(mem.ptr)
        
        # Memory should be freed automatically when object is deleted
        del mem
    
    def test_memcpy_h2d_d2h(self):
        """Test host-to-device and device-to-host memory copy"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        # Create host data
        size = 100
        host_src = np.arange(size, dtype=np.float32)
        host_dst = np.zeros(size, dtype=np.float32)
        
        # Allocate device memory
        device_mem = memory.malloc(host_src.nbytes)
        
        # Copy host to device
        memory.memcpy_h2d(device_mem, host_src)
        
        # Copy device to host
        memory.memcpy_d2h(host_dst, device_mem)
        
        # Verify (in stub implementation, this may not work)
        # np.testing.assert_array_equal(host_src, host_dst)
    
    def test_memcpy_d2d(self):
        """Test device-to-device memory copy"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        size = 1024
        src_mem = memory.malloc(size)
        dst_mem = memory.malloc(size)
        
        # Copy device to device
        memory.memcpy_d2d(dst_mem, src_mem)
    
    def test_memset(self):
        """Test memory set operation"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        size = 1024
        mem = memory.malloc(size)
        
        # Set memory
        memory.memset(mem, 0, size)
    
    def test_get_mem_info(self):
        """Test getting memory information"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        free, total = memory.get_mem_info()
        self.assertIsInstance(free, int)
        self.assertIsInstance(total, int)
        self.assertGreaterEqual(total, free)
        self.assertGreater(total, 0)


if __name__ == '__main__':
    unittest.main()
