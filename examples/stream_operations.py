"""
Example: Stream and event usage

This example demonstrates asynchronous stream operations and
event-based timing on Iluvatar GPU.
"""

import time
import paddle_iluvatar.device as device
import paddle_iluvatar.stream as stream


def main():
    print("=" * 60)
    print("Iluvatar GPU Stream and Event Operations")
    print("=" * 60)
    
    # Check if GPU is available
    if not device.is_available():
        print("No Iluvatar GPU available!")
        return
    
    # Create streams
    print("\nCreating streams...")
    stream1 = stream.Stream()
    stream2 = stream.Stream()
    print(f"Stream 1: {stream1.handle}")
    print(f"Stream 2: {stream2.handle}")
    
    # Create events
    print("\nCreating events...")
    start_event = stream.Event()
    end_event = stream.Event()
    print("Events created!")
    
    # Record start event
    print("\nRecording start event...")
    start_event.record(stream1)
    
    # Simulate some work
    print("Simulating GPU work...")
    time.sleep(0.1)  # In real scenario, this would be GPU operations
    
    # Record end event
    print("Recording end event...")
    end_event.record(stream1)
    
    # Synchronize stream
    print("\nSynchronizing stream...")
    stream1.synchronize()
    
    # Measure elapsed time
    elapsed = start_event.elapsed_time(end_event)
    print(f"Elapsed time: {elapsed:.3f} ms")
    
    # Query stream status
    print("\nQuerying stream status...")
    completed = stream1.query()
    print(f"Stream 1 completed: {completed}")
    
    # Wait for stream
    print("\nWaiting for stream 2...")
    stream2.wait()
    print("Stream 2 completed!")
    
    # Clean up
    print("\nCleaning up...")
    del stream1, stream2
    del start_event, end_event
    print("Done!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
