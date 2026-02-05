cmake_minimum_required(VERSION 3.10)
project(paddle_iluvatar VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Iluvatar SDK (when available)
# set(ILUVATAR_SDK_PATH $ENV{ILUVATAR_SDK_PATH})
# if(NOT ILUVATAR_SDK_PATH)
#     message(WARNING "ILUVATAR_SDK_PATH not set. Using stub implementation.")
# endif()

# Source files
set(SOURCE_FILES
    csrc/device.cpp
    csrc/memory.cpp
    csrc/stream.cpp
    csrc/error.cpp
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create library
add_library(paddle_iluvatar SHARED ${SOURCE_FILES})

# Install targets
install(TARGETS paddle_iluvatar
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
