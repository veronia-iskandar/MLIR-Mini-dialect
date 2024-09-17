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
    // Prevent tiling an operation that's already tiled.
    if (matMulOp->hasAttr("tiled")) {
      llvm::outs() << "MatMulOp is already tiled. Skipping.\n";
      return failure(); // Return failure if it's already tiled.
    }

    // Add debug output to check the location of the operation
    llvm::outs() << "Tiling MatMulOp at location: " << matMulOp.getLoc() << "\n";

    // Get the location and operands
    Location loc = matMulOp.getLoc();
    Value lhs = matMulOp.getOperand(0);
    Value rhs = matMulOp.getOperand(1);

    // Tile size for the matmul (e.g., 8x8)
    SmallVector<int64_t, 2> tileSizes = {8, 8};

    // Add debug output to show the tile size being used
    llvm::outs() << "Using tile size: " << tileSizes[0] << "x" << tileSizes[1] << "\n";

    // Get the shape of the input operands (assuming they're 2D tensors)
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    auto rhsType = rhs.getType().cast<RankedTensorType>();

    if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
      llvm::outs() << "Operand ranks are not 2D tensors, aborting tiling.\n";
      return failure();
    }

    // Tile sizes as Values
    SmallVector<Value, 2> sizes = {
        rewriter.create<arith::ConstantIndexOp>(loc, tileSizes[0]),
        rewriter.create<arith::ConstantIndexOp>(loc, tileSizes[1])
    };

    // Strides (set to 1 for dense tiling)
    SmallVector<Value, 2> strides = {
        rewriter.create<arith::ConstantIndexOp>(loc, 1),
        rewriter.create<arith::ConstantIndexOp>(loc, 1)
    };

    // Define offsets (i.e., starting points for slicing), which can be dynamic based on loop indices
    SmallVector<Value, 2> offsets = {
        rewriter.create<arith::ConstantIndexOp>(loc, 0),
        rewriter.create<arith::ConstantIndexOp>(loc, 0)
    };

    // Debug output before creating the slices
    llvm::outs() << "Creating slices for LHS and RHS operands...\n";

    // Extract a slice of the LHS operand (a submatrix)
    Value lhsSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, lhs, offsets, sizes, strides);

    // Extract a slice of the RHS operand (a submatrix)
    Value rhsSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, rhs, offsets, sizes, strides);

    // Create a new tiled MatMul operation using the slices
    //auto resultType = matMulOp.getType().cast<RankedTensorType>();
    //Value resultTile = rewriter.create<MatMulOp>(loc, resultType, lhsSlice, rhsSlice);
    
    auto resultType = RankedTensorType::get({tileSizes[0], tileSizes[1]}, lhsType.getElementType());
    Value resultTile = rewriter.create<MatMulOp>(loc, resultType, lhsSlice, rhsSlice);


    // Mark the **newly created tiled MatMulOp** as tiled to prevent it from being tiled again
    resultTile.getDefiningOp()->setAttr("tiled", rewriter.getUnitAttr());

    // Mark the original MatMulOp as tiled to prevent recursive tiling
    matMulOp->setAttr("tiled", rewriter.getUnitAttr());

    // Debug output to confirm the replacement of the original operation
    llvm::outs() << "Replacing original MatMulOp with tiled version.\n";

    // Replace the original matmul with the tiled version
    rewriter.replaceOp(matMulOp, resultTile);

    return success();
  }
};

struct TileAddOpPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
    Location loc = addOp.getLoc();

    // Check if the operation has already been tiled
    if (addOp->hasAttr("tiled")) {
      llvm::outs() << "AddOp is already tiled. Skipping.\n";
      return failure();
    }

    // Access the LHS and RHS operands
    Value lhs = addOp.getOperand(0);
    Value rhs = addOp.getOperand(1);

    // Get the types of the LHS and RHS operands
    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();

    if (!lhsType || !rhsType) {
      llvm::errs() << "Expected ranked tensor types for AddOp operands.\n";
      return failure();
    }

    // Ensure the LHS and RHS shapes are compatible
    if (lhsType.getShape() != rhsType.getShape()) {
      llvm::errs() << "LHS and RHS shapes must match for tiling AddOp.\n";
      return failure();
    }

    // Check if the result tensor has static sizes
    auto resultShape = lhsType.getShape();
    if (!llvm::all_of(resultShape, [](int64_t sz) { return !ShapedType::isDynamic(sz); })) {
      llvm::errs() << "Error: Tiling expects static sizes, but dynamic sizes were encountered.\n";
      return failure();
    }

    // Tile sizes: we assume an 8x8 tiling for now
    SmallVector<int64_t, 2> tileSizes = {8, 8};

    // Convert static integers to OpFoldResult using rewriter.getIndexAttr
    SmallVector<OpFoldResult, 2> offsets = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult, 2> sizes = {rewriter.getIndexAttr(tileSizes[0]), rewriter.getIndexAttr(tileSizes[1])};
    SmallVector<OpFoldResult, 2> strides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

    // Extract slices from the LHS and RHS operands using static values
    Value lhsSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, lhs, offsets, sizes, strides);
    Value rhsSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, rhs, offsets, sizes, strides);

    // Create a new AddOp for the tiled slices
    auto resultType = RankedTensorType::get(tileSizes, lhsType.getElementType());
    Value tiledAddOp = rewriter.create<AddOp>(loc, resultType, lhsSlice, rhsSlice);

    // Insert the tiled result back into the larger destination tensor (of size 16x16)
    auto destTensorType = RankedTensorType::get(resultShape, lhsType.getElementType());
    Value destTensor = rewriter.create<tensor::EmptyOp>(loc, resultShape, lhsType.getElementType());

    // Insert the result back into the destination tensor using static values
    Value tiledResult = rewriter.create<tensor::InsertSliceOp>(loc, tiledAddOp, destTensor,
                                                               offsets, sizes, strides);

    // Replace the original AddOp with the tiled result
    rewriter.replaceOp(addOp, tiledResult);

    // Mark the operation as "tiled" to prevent re-tiling
    tiledAddOp.getDefiningOp()->setAttr("tiled", rewriter.getUnitAttr());

    llvm::outs() << "Tiling AddOp at location: " << addOp.getLoc() << "\n";
    llvm::outs() << "Using tile size: " << tileSizes[0] << "x" << tileSizes[1] << "\n";

    return success();
  }
};









} // namespace

/// Definition of the runOnOperation function
void TileMiniOpsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *context = &getContext();

  llvm::dbgs() << "Running tiling pass on function: " << func.getName() << "\n";
  
  // Add tiling patterns for MatMulOp and AddOp
  RewritePatternSet patterns(context);
  patterns.add<TileMatMulOpPattern, TileAddOpPattern>(context);

  llvm::dbgs() << "Applying tiling patterns.\n";
  // Apply patterns to the function
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    llvm::dbgs() << "Tiling patterns application failed.\n";
    signalPassFailure();
  }
  llvm::dbgs() << "Successfully applied tiling patterns.\n";
}

/// Factory function to create the pass
std::unique_ptr<OperationPass<mlir::func::FuncOp>> mlir::mini::createTileMiniOpsPass() {
  return std::make_unique<TileMiniOpsPass>();
}

