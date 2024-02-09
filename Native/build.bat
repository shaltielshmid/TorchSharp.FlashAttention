if not exist "build-win" (
    rmdir /s /q build-win
)
mkdir build-win
cd build-win
echo -D%1=%2 -D%3=%4 -D%5=%6
cmake ..\LibFlashAttention -D%1=%2 -D%3=%4 -D%5=%6
cmake --build . --config Release
