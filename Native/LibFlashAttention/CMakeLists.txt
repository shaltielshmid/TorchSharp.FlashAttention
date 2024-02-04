cmake_minimum_required(VERSION 3.18)

project(LibFlashAttention)

# Path to CUDA
find_package(CUDA REQUIRED)

# Path to LibTorch
find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})

# Include directories
include_directories(
    ${FLASH_PATH}/csrc/flash_attn
    ${FLASH_PATH}/csrc/flash_attn/src
    ${FLASH_PATH}/csrc/cutlass/include
    ${LIBTORCH_PATH}/include
    ${LIBTORCH_PATH}/include/torch/csrc/api/include
    ${LIBTORCH_PATH}/include/torch/csrc/api/include/torch
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    LibFlashAttention.cpp
)

# Header files
set(HEADERS
    LibFlashAttention.h
    Utils.h
    Stdafx.h
    UnixSal.h
)

# Define the library target
add_library(LibFlashAttention SHARED ${SOURCES} ${HEADERS})

# Link libraries

target_link_libraries(LibFlashAttention
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${FLASH_PATH}/../../compiled-runtimes/linux-x64/native/libflash_attn.so
)

# Set properties
set_property(TARGET LibFlashAttention PROPERTY CXX_STANDARD 17)
