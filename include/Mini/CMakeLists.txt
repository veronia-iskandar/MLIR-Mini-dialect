# Define LLVM target definitions for the MiniOps
set(LLVM_TARGET_DEFINITIONS MiniOps.td)

# Generate operation declarations and definitions for MiniOps
mlir_tablegen(MiniOps.h.inc -gen-op-decls -name MiniOps)
mlir_tablegen(MiniOps.cpp.inc -gen-op-defs -name MiniOps)

# Generate dialect declarations and definitions for the MiniDialect
mlir_tablegen(MiniOpsDialect.h.inc -gen-dialect-decls -name MiniDialect)
mlir_tablegen(MiniOpsDialect.cpp.inc -gen-dialect-defs -name MiniDialect)

# Add the TableGen target for MiniOps
add_public_tablegen_target(MLIRMiniOpsIncGen)

# Ensure TilingInterface and SideEffectInterface are included
mlir_tablegen(TilingInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(TilingInterface.cpp.inc -gen-op-interface-defs)

mlir_tablegen(SideEffectInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SideEffectInterfaces.cpp.inc -gen-op-interface-defs)

# Add the public tablegen target for tiling and side-effect interfaces
add_public_tablegen_target(MLIRTilingInterfaceIncGen)
#add_public_tablegen_target(MLIRSideEffectInterfacesIncGen)

# No cyclic dependencies here, just generate the targets



# Define LLVM target definitions for ShapeInferenceInterface
#set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
#set(LLVM_TARGET_DEFINITIONS TilingInterface.td)

# Generate the interface declarations and definitions for ShapeInferenceInterface
#mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls -name ShapeInferenceInterface)
#mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs -name ShapeInferenceInterface)
#mlir_tablegen(TilingInterface.h.inc -gen-op-interface-decls -name TilingInterface)
#mlir_tablegen(TilingInterface.cpp.inc -gen-op-interface-defs -name TilingInterface)



# Add the tablegen target for ShapeInferenceInterface
#add_public_tablegen_target(MLIRShapeInferenceInterfaceIncGen)
#add_public_tablegen_target(MLIRTilingInterfaceIncGen)


