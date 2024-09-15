get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_llvm_executable(mini-opt mini-opt.cpp)

target_link_libraries(mini-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MiniDialect
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRMlirOptMain
  MLIRDialect
  MLIRFuncDialect
  MLIRArithDialect
)

mlir_check_all_link_libraries(mini-opt)

