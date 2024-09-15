add_library(MiniDialect
  MiniDialect.cpp
  MiniOps.cpp
)

target_include_directories(MiniDialect PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

add_dependencies(MiniDialect
  MLIRMiniOpsIncGen
)

target_link_libraries(MiniDialect
  MLIRIR
  MLIRSideEffectInterfaces  # Include required MLIR libraries
)

