#include "paddle_iluvatar/device.h"
#include "paddle_iluvatar/error.h"
#include <cstring>
#include <vector>

namespace paddle_iluvatar {

// Global device state
static bool g_initialized = false;
static int g_current_device = 0;
static int g_device_count = 0;

int Device::GetDeviceCount() {
    if (!g_initialized) {
        Initialize();
    }
    return g_device_count;
}

void Device::SetDevice(int device_id) {
    if (!g_initialized) {
        Initialize();
    }
    if (device_id < 0 || device_id >= g_device_count) {
        throw Error(ERROR_INVALID_DEVICE, 
                   "Invalid device ID: " + std::to_string(device_id));
    }
    g_current_device = device_id;
}

int Device::GetDevice() {
    if (!g_initialized) {
        Initialize();
    }
    return g_current_device;
}

void Device::GetDeviceProperties(DeviceProperties* props, int device_id) {
    if (!g_initialized) {
        Initialize();
    }
    if (device_id < 0 || device_id >= g_device_count) {
        throw Error(ERROR_INVALID_DEVICE, 
                   "Invalid device ID: " + std::to_string(device_id));
    }
    
    // Stub implementation - populate with actual Iluvatar GPU properties
    std::strcpy(props->name, "Iluvatar GPU");
    props->total_memory = 16ULL * 1024 * 1024 * 1024; // 16GB
    props->compute_capability_major = 1;
    props->compute_capability_minor = 0;
    props->multi_processor_count = 80;
    props->max_threads_per_block = 1024;
    props->max_threads_per_multi_processor = 2048;
}

void Device::Synchronize() {
    if (!g_initialized) {
        Initialize();
    }
    // Stub: Wait for all operations on current device to complete
    // In actual implementation, this would call Iluvatar SDK's sync function
}

void Device::Initialize() {
    if (g_initialized) {
        return;
    }
    
    // Stub: Initialize Iluvatar runtime
    // In actual implementation, this would:
    // 1. Load Iluvatar driver
    // 2. Enumerate available devices
    // 3. Initialize device contexts
    
    // For now, simulate one device
    g_device_count = 1;
    g_current_device = 0;
    g_initialized = true;
}

void Device::Finalize() {
    if (!g_initialized) {
        return;
    }
    
    // Stub: Clean up Iluvatar runtime
    // In actual implementation, this would:
    // 1. Release device contexts
    // 2. Unload driver
    
    g_initialized = false;
    g_device_count = 0;
    g_current_device = 0;
}

} // namespace paddle_iluvatar
