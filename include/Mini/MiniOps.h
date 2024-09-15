//===- MiniOps.h - Mini Operations ------------------------------*- C++ -*-===//
//
// This file declares the operations for the Mini dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MINI_MINIOPS_H
#define MINI_MINIOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Mini/MiniOps.h.inc"

#endif // MINI_MINIOPS_H

