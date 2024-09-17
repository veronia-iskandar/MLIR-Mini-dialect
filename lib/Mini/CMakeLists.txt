# Create the MiniDialect library
add_library(MiniDialect
  MiniDialect.cpp
  MiniOps.cpp
  TileMiniOpsPass.cpp
  Passes.cpp
)

# Ensure that the generated headers are found during compilation
target_include_directories(MiniDialect PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${MLIR_BINARY_DIR}/include
  ${MLIR_SOURCE_DIR}/include
)

# Add the TableGen dependencies here (no circular dependencies)
add_dependencies(MiniDialect
  MLIRMiniOpsIncGen
)

# Link the required MLIR components for your dialect
target_link_libraries(MiniDialect
  MLIRIR                        # Core MLIR functionality
  MLIRLinalgDialect                    # Linalg dialect for tiling
  MLIRSCFDialect                       # SCF dialect
  MLIRTransforms                # Transforms like tiling
  MLIRFuncDialect                      # Func dialect
  MLIRTilingInterface           # Tiling interface
  MLIRBufferization 
)

