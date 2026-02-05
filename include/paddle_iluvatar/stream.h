#ifndef PADDLE_ILUVATAR_STREAM_H_
#define PADDLE_ILUVATAR_STREAM_H_

namespace paddle_iluvatar {

// Forward declaration
typedef void* StreamHandle;

// Stream management functions
class Stream {
public:
    // Create a stream
    static StreamHandle Create();
    
    // Destroy a stream
    static void Destroy(StreamHandle stream);
    
    // Synchronize a stream
    static void Synchronize(StreamHandle stream);
    
    // Query stream status
    static bool Query(StreamHandle stream);
    
    // Wait for stream completion
    static void Wait(StreamHandle stream);
};

// Event management functions
class Event {
public:
    typedef void* EventHandle;
    
    // Create an event
    static EventHandle Create();
    
    // Destroy an event
    static void Destroy(EventHandle event);
    
    // Record event on stream
    static void Record(EventHandle event, StreamHandle stream);
    
    // Synchronize event
    static void Synchronize(EventHandle event);
    
    // Query event status
    static bool Query(EventHandle event);
    
    // Measure elapsed time between events
    static float ElapsedTime(EventHandle start, EventHandle end);
};

} // namespace paddle_iluvatar

#endif // PADDLE_ILUVATAR_STREAM_H_
