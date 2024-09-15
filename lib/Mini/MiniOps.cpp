//===- MiniOps.cpp - Mini Operations Implementation -------------*- C++ -*-===//
//
// This file implements the operations for the Mini dialect.
//
//===----------------------------------------------------------------------===//

#include "Mini/MiniOps.h"
#include "Mini/MiniDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h" 

using namespace mlir;
using namespace mlir::mini;

#define GET_OP_CLASSES
#include "Mini/MiniOps.cpp.inc"

