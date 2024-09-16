//===- MiniDialect.cpp - Mini Dialect Implementation ------------*- C++ -*-===//
//
// This file implements the Mini dialect.
//
//===----------------------------------------------------------------------===//

#include "Mini/MiniDialect.h"
#include "Mini/MiniOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
using namespace mlir;
using namespace mlir::mini;

#include "Mini/MiniOpsDialect.cpp.inc"

void MiniDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mini/MiniOps.cpp.inc"
      >();
  // Load the required dialects
  getContext()->getOrLoadDialect<mlir::tensor::TensorDialect>();
  getContext()->getOrLoadDialect<mlir::arith::ArithDialect>();
}


