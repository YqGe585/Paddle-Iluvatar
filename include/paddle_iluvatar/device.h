#ifndef PADDLE_ILUVATAR_DEVICE_H_
#define PADDLE_ILUVATAR_DEVICE_H_

#include <cstdint>
#include <string>

namespace paddle_iluvatar {

// Device properties
struct DeviceProperties {
    char name[256];
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multi_processor_count;
    int max_threads_per_block;
    int max_threads_per_multi_processor;
};

// Device management functions
class Device {
public:
    // Get number of available devices
    static int GetDeviceCount();
    
    // Set current device
    static void SetDevice(int device_id);
    
    // Get current device ID
    static int GetDevice();
    
    // Get device properties
    static void GetDeviceProperties(DeviceProperties* props, int device_id);
    
    // Synchronize device
    static void Synchronize();
    
    // Initialize device
    static void Initialize();
    
    // Finalize device
    static void Finalize();
};

} // namespace paddle_iluvatar

#endif // PADDLE_ILUVATAR_DEVICE_H_
