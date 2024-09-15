# Clean the build directory
rm -rf build
mkdir build
cd build

# Run CMake
cmake -G Ninja .. \
  -DMLIR_DIR=/home/veronia/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/veronia/llvm-project/build/lib/cmake/llvm

# Build the project
ninja

