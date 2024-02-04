if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build
cmake ../LibFlashAttention -DLIBTORCH_PATH=$LIBTORCH_PATH -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH -DFLASH_PATH=$FLASH_PATH
cmake --build . --config Release
