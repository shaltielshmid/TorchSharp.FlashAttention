cd flash-attention
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build
cmake .. -DLIBTORCH_PATH=../../libtorch-cuda-12.1/libtorch-cuda-12.1/libtorch-shared-with-deps/libtorch -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR
cmake --build . --config Release
cp libflash_attn.so ../../../compiled-runtimes/linux-x64/native/libflash_attn.so