cd flash-attention
if exist "build" (
    rmdir /s /q build
)
mkdir build
cd build
REM Add these flags to the next line if using visual studio >= 17.40: -DCUDA_NVCC_FLAGS="--allow-unsupported-compiler" -DCMAKE_CXX_FLAGS="/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
cmake .. -DLIBTORCH_PATH=..\..\libtorch-cuda-12.1\libtorch-cuda-12.1\libtorch-win-shared-with-deps\libtorch 
cmake --build . --config Release --target ALL_BUILD
copy Release\* ..\..\..\compiled-runtimes\win-x64\native\