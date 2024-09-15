set(LLVM_TARGET_DEFINITIONS MiniOps.td)

mlir_tablegen(MiniOps.h.inc -gen-op-decls -name MiniOps)
mlir_tablegen(MiniOps.cpp.inc -gen-op-defs -name MiniOps)
mlir_tablegen(MiniOpsDialect.h.inc -gen-dialect-decls -name MiniDialect)
mlir_tablegen(MiniOpsDialect.cpp.inc -gen-dialect-defs -name MiniDialect)

add_public_tablegen_target(MLIRMiniOpsIncGen)

