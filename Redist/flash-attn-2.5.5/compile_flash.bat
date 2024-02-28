cd flash-attention
if exist "build" (
    rmdir /s /q build
)
mkdir build
cd build
cmake .. -DLIBTORCH_PATH=..\..\libtorch-cuda-12.1\libtorch-cuda-12.1\libtorch-win-shared-with-deps\libtorch
cmake --build . --config Release --target ALL_BUILD
copy Release\* ..\..\..\compiled-runtimes\win-x64\native\