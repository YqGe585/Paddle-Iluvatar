#ifndef PADDLE_ILUVATAR_ERROR_H_
#define PADDLE_ILUVATAR_ERROR_H_

#include <string>
#include <stdexcept>

namespace paddle_iluvatar {

// Error codes
enum ErrorCode {
    SUCCESS = 0,
    ERROR_INVALID_VALUE = 1,
    ERROR_OUT_OF_MEMORY = 2,
    ERROR_NOT_INITIALIZED = 3,
    ERROR_DEVICE_NOT_FOUND = 4,
    ERROR_INVALID_DEVICE = 5,
    ERROR_MEMORY_COPY_FAILED = 6,
    ERROR_LAUNCH_FAILED = 7,
    ERROR_SYNCHRONIZATION_FAILED = 8,
    ERROR_UNKNOWN = 999
};

// Error handling
class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}
    
    ErrorCode code() const { return code_; }
    
private:
    ErrorCode code_;
};

// Get last error
ErrorCode GetLastError();

// Get error string
std::string GetErrorString(ErrorCode code);

// Check and throw on error
void CheckError(ErrorCode code, const char* file, int line);

// Macro for error checking
#define PADDLE_ILUVATAR_CHECK(call) \
    do { \
        paddle_iluvatar::ErrorCode error = (call); \
        if (error != paddle_iluvatar::SUCCESS) { \
            paddle_iluvatar::CheckError(error, __FILE__, __LINE__); \
        } \
    } while(0)

} // namespace paddle_iluvatar

#endif // PADDLE_ILUVATAR_ERROR_H_
