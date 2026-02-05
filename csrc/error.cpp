#include "paddle_iluvatar/error.h"
#include <sstream>

namespace paddle_iluvatar {

// Global error state
static ErrorCode g_last_error = SUCCESS;

ErrorCode GetLastError() {
    return g_last_error;
}

std::string GetErrorString(ErrorCode code) {
    switch (code) {
        case SUCCESS:
            return "Success";
        case ERROR_INVALID_VALUE:
            return "Invalid value";
        case ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case ERROR_DEVICE_NOT_FOUND:
            return "Device not found";
        case ERROR_INVALID_DEVICE:
            return "Invalid device";
        case ERROR_MEMORY_COPY_FAILED:
            return "Memory copy failed";
        case ERROR_LAUNCH_FAILED:
            return "Kernel launch failed";
        case ERROR_SYNCHRONIZATION_FAILED:
            return "Synchronization failed";
        case ERROR_UNKNOWN:
        default:
            return "Unknown error";
    }
}

void CheckError(ErrorCode code, const char* file, int line) {
    if (code != SUCCESS) {
        g_last_error = code;
        std::ostringstream oss;
        oss << "Iluvatar error at " << file << ":" << line 
            << " - " << GetErrorString(code);
        throw Error(code, oss.str());
    }
}

} // namespace paddle_iluvatar
