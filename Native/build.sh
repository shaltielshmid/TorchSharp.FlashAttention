if [ -d "build-linux" ]; then
    rm -rf build-linux
fi
mkdir build-linux
cd build-linux
cmake ../LibFlashAttention -DLIBTORCH_PATH=$LIBTORCH_PATH -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH -DFLASH_PATH=$FLASH_PATH
cmake --build . --config Release
