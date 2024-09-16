//===- MiniOps.h - Mini Operations ------------------------------*- C++ -*-===//
//
// This file declares the operations for the Mini dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MINI_MINIOPS_H
#define MINI_MINIOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
//#include "/home/veronia/llvm-project/build/tools/mlir/include/mlir/Interfaces/TilingInterface.h.inc"
//#include "TilingInterface.h"
//#include "ShapeInferenceInterface.h"
#define GET_OP_CLASSES
#include "Mini/MiniOps.h.inc"

#endif // MINI_MINIOPS_H

