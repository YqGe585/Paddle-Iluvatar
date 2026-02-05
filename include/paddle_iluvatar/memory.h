#ifndef PADDLE_ILUVATAR_MEMORY_H_
#define PADDLE_ILUVATAR_MEMORY_H_

#include <cstddef>

namespace paddle_iluvatar {

// Memory management functions
class Memory {
public:
    // Allocate device memory
    static void* Malloc(size_t size);
    
    // Free device memory
    static void Free(void* ptr);
    
    // Copy memory from host to device
    static void MemcpyH2D(void* dst, const void* src, size_t size);
    
    // Copy memory from device to host
    static void MemcpyD2H(void* dst, const void* src, size_t size);
    
    // Copy memory from device to device
    static void MemcpyD2D(void* dst, const void* src, size_t size);
    
    // Set device memory
    static void Memset(void* ptr, int value, size_t size);
    
    // Get memory info
    static void GetMemInfo(size_t* free, size_t* total);
};

} // namespace paddle_iluvatar

#endif // PADDLE_ILUVATAR_MEMORY_H_
