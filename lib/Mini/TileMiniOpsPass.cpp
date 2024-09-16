#include "Mini/TileMiniOpsPass.h"
#include "Mini/MiniOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::mini;

namespace {
struct TileMatMulOpPattern : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp matMulOp, PatternRewriter &rewriter) const override {
    // Tiling options using linalg
    mlir::linalg::LinalgTilingOptions tilingOptions;
    tilingOptions.setTileSizes({8, 8});

    // Try tiling using Linalg tiling functions.
    auto linalgOp = dyn_cast_or_null<mlir::linalg::LinalgOp>(matMulOp.getOperation());
    if (!linalgOp)
      return failure();

    // Tile the operation using linalg tiling
    FailureOr<mlir::linalg::TiledLinalgOp> tiledOp =
        mlir::linalg::tileLinalgOp(rewriter, linalgOp, tilingOptions);

    if (failed(tiledOp))
      return failure();

    // Replace the original operation with the tiled results
    rewriter.replaceOp(matMulOp, tiledOp->tensorResults);
    return success();
  }
};

struct TileAddOpPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
    // Tiling options using linalg
    mlir::linalg::LinalgTilingOptions tilingOptions;
    tilingOptions.setTileSizes({8, 8});

    // Try tiling using Linalg tiling functions.
    auto linalgOp = dyn_cast_or_null<mlir::linalg::LinalgOp>(addOp.getOperation());
    if (!linalgOp)
      return failure();

    // Tile the operation using linalg tiling
    FailureOr<mlir::linalg::TiledLinalgOp> tiledOp =
        mlir::linalg::tileLinalgOp(rewriter, linalgOp, tilingOptions);

    if (failed(tiledOp))
      return failure();

    // Replace the original operation with the tiled results
    rewriter.replaceOp(addOp, tiledOp->tensorResults);
    return success();
  }
    
};
} // namespace

/// Definition of the runOnOperation function
void TileMiniOpsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *context = &getContext();

  // Add tiling patterns for MatMulOp and AddOp
  RewritePatternSet patterns(context);
  patterns.add<TileMatMulOpPattern, TileAddOpPattern>(context);

  // Apply patterns to the function
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
    signalPassFailure();
}

/// Factory function to create the pass
std::unique_ptr<OperationPass<mlir::func::FuncOp>> mlir::mini::createTileMiniOpsPass() {
  return std::make_unique<TileMiniOpsPass>();
}

