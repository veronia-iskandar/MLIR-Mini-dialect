#ifndef MINI_PASSES_H
#define MINI_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // Include for func::FuncOp

namespace mlir {
namespace mini {

//std::unique_ptr<OperationPass<func::FuncOp>> createTileMiniOpsPass();

void registerTileMiniOpsPass();

} // namespace mini
} // namespace mlir

#endif // MINI_PASSES_H

