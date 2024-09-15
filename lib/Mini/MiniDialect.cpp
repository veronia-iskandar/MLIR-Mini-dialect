//===- MiniDialect.cpp - Mini Dialect Implementation ------------*- C++ -*-===//
//
// This file implements the Mini dialect.
//
//===----------------------------------------------------------------------===//

#include "Mini/MiniDialect.h"
#include "Mini/MiniOps.h"

using namespace mlir;
using namespace mlir::mini;

#include "Mini/MiniOpsDialect.cpp.inc"

void MiniDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mini/MiniOps.cpp.inc"
      >();
}

