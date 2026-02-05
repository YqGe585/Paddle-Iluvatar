#include "paddle_iluvatar/memory.h"
#include "paddle_iluvatar/error.h"
#include <cstdlib>
#include <cstring>
#include <map>

namespace paddle_iluvatar {

// Simple memory allocator stub
// In production, this would use Iluvatar's memory management API
static std::map<void*, size_t> g_allocations;

void* Memory::Malloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    
    // Stub: Allocate device memory
    // In actual implementation: iluDeviceMalloc(&ptr, size) or similar
    void* ptr = std::malloc(size);
    if (!ptr) {
        throw Error(ERROR_OUT_OF_MEMORY, 
                   "Failed to allocate " + std::to_string(size) + " bytes");
    }
    
    g_allocations[ptr] = size;
    return ptr;
}

void Memory::Free(void* ptr) {
    if (!ptr) {
        return;
    }
    
    // Stub: Free device memory
    // In actual implementation: iluDeviceFree(ptr) or similar
    auto it = g_allocations.find(ptr);
    if (it == g_allocations.end()) {
        throw Error(ERROR_INVALID_VALUE, "Invalid memory pointer");
    }
    
    std::free(ptr);
    g_allocations.erase(it);
}

void Memory::MemcpyH2D(void* dst, const void* src, size_t size) {
    if (size == 0) {
        return;
    }
    if (!dst || !src) {
        throw Error(ERROR_INVALID_VALUE, "Null pointer in memcpy");
    }
    
    // Stub: Copy from host to device
    // In actual implementation: iluMemcpyH2D(dst, src, size) or similar
    std::memcpy(dst, src, size);
}

void Memory::MemcpyD2H(void* dst, const void* src, size_t size) {
    if (size == 0) {
        return;
    }
    if (!dst || !src) {
        throw Error(ERROR_INVALID_VALUE, "Null pointer in memcpy");
    }
    
    // Stub: Copy from device to host
    // In actual implementation: iluMemcpyD2H(dst, src, size) or similar
    std::memcpy(dst, src, size);
}

void Memory::MemcpyD2D(void* dst, const void* src, size_t size) {
    if (size == 0) {
        return;
    }
    if (!dst || !src) {
        throw Error(ERROR_INVALID_VALUE, "Null pointer in memcpy");
    }
    
    // Stub: Copy from device to device
    // In actual implementation: iluMemcpyD2D(dst, src, size) or similar
    std::memcpy(dst, src, size);
}

void Memory::Memset(void* ptr, int value, size_t size) {
    if (size == 0) {
        return;
    }
    if (!ptr) {
        throw Error(ERROR_INVALID_VALUE, "Null pointer in memset");
    }
    
    // Stub: Set device memory
    // In actual implementation: iluMemset(ptr, value, size) or similar
    std::memset(ptr, value, size);
}

void Memory::GetMemInfo(size_t* free, size_t* total) {
    // Stub: Get memory information
    // In actual implementation: iluMemGetInfo(free, total) or similar
    *total = 16ULL * 1024 * 1024 * 1024; // 16GB
    
    size_t allocated = 0;
    for (const auto& kv : g_allocations) {
        allocated += kv.second;
    }
    *free = *total - allocated;
}

} // namespace paddle_iluvatar
