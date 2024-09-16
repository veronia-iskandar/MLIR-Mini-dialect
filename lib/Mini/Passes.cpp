#include "Mini/Passes.h"
#include "Mini/TileMiniOpsPass.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mini {

//std::unique_ptr<OperationPass<func::FuncOp>> createTileMiniOpsPass() {
  //return std::make_unique<TileMiniOpsPass>();
//}

void registerTileMiniOpsPass() {
  PassRegistration<TileMiniOpsPass>(
      []() -> std::unique_ptr<Pass> { return createTileMiniOpsPass(); });
}

} // namespace mini
} // namespace mlir

