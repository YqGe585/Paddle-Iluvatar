"""
Unit tests for stream management
"""

import unittest
import paddle_iluvatar.stream as stream
import paddle_iluvatar.device as device


class TestStream(unittest.TestCase):
    
    def test_stream_create_destroy(self):
        """Test stream creation and destruction"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        s = stream.Stream()
        self.assertIsNotNone(s)
        self.assertIsNotNone(s.handle)
        
        # Should be destroyed automatically
        del s
    
    def test_stream_synchronize(self):
        """Test stream synchronization"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        s = stream.Stream()
        
        # Should not raise an exception
        s.synchronize()
        s.wait()
    
    def test_stream_query(self):
        """Test stream query"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        s = stream.Stream()
        completed = s.query()
        self.assertIsInstance(completed, bool)


class TestEvent(unittest.TestCase):
    
    def test_event_create_destroy(self):
        """Test event creation and destruction"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        e = stream.Event()
        self.assertIsNotNone(e)
        self.assertIsNotNone(e.handle)
        
        # Should be destroyed automatically
        del e
    
    def test_event_record(self):
        """Test event recording"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        s = stream.Stream()
        e = stream.Event()
        
        # Should not raise an exception
        e.record(s)
    
    def test_event_synchronize(self):
        """Test event synchronization"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        e = stream.Event()
        
        # Should not raise an exception
        e.synchronize()
    
    def test_event_query(self):
        """Test event query"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        e = stream.Event()
        completed = e.query()
        self.assertIsInstance(completed, bool)
    
    def test_event_elapsed_time(self):
        """Test measuring elapsed time between events"""
        if not device.is_available():
            self.skipTest("No Iluvatar GPU available")
        
        s = stream.Stream()
        start = stream.Event()
        end = stream.Event()
        
        start.record(s)
        # Do some work here
        end.record(s)
        
        elapsed = start.elapsed_time(end)
        self.assertIsInstance(elapsed, float)
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == '__main__':
    unittest.main()
