#ifndef MINI_TILEMINIOPSPASS_H
#define MINI_TILEMINIOPSPASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace mini {

/// Pass that tiles mini.matmul and mini.add operations.
struct TileMiniOpsPass : public PassWrapper<TileMiniOpsPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override;
      // Override getArgument() to provide a unique identifier for the pass
  StringRef getArgument() const final { return "tile-mini-ops"; }

  // Override getDescription() to describe what this pass does
  StringRef getDescription() const final {
    return "Tile the matmul and add ops in the Mini dialect.";
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createTileMiniOpsPass();

} // namespace mini
} // namespace mlir

#endif // MINI_TILEMINIOPSPASS_H

