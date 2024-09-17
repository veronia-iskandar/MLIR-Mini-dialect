//===- MiniOps.cpp - Mini Operations Implementation -------------*- C++ -*-===//
//
// This file implements the operations for the Mini dialect.
//
//===----------------------------------------------------------------------===//

#include "Mini/MiniOps.h"
#include "Mini/MiniDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h" 
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"



using namespace mlir;
using namespace mlir::mini;
using namespace mlir::bufferization;
#define GET_OP_CLASSES
#include "Mini/MiniOps.cpp.inc"

// Helper function to extract constant integer values
static int64_t getConstantIntValue(OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>())
    return attr.cast<IntegerAttr>().getInt();
  llvm_unreachable("Expected constant integer value");
}

//===----------------------------------------------------------------------===//
// MatMulOp: Tiling Interface Methods
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> MatMulOp::getLoopIteratorTypes() {
  return {
    utils::IteratorType::parallel,  // Loop over M
    utils::IteratorType::parallel,  // Loop over N
    utils::IteratorType::reduction  // Loop over K
  };
}

SmallVector<Range> MatMulOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  SmallVector<Range> loopRanges;

  Value lhs = getLhs();
  Value rhs = getRhs();

  Value m = builder.create<tensor::DimOp>(loc, lhs, 0);
  Value n = builder.create<tensor::DimOp>(loc, rhs, 1);
  Value k = builder.create<tensor::DimOp>(loc, lhs, 1);

  OpFoldResult lowerBound = builder.getIndexAttr(0);
  OpFoldResult step = builder.getIndexAttr(1);

  loopRanges.push_back(Range{lowerBound, m, step});
  loopRanges.push_back(Range{lowerBound, n, step});
  loopRanges.push_back(Range{lowerBound, k, step});

  return loopRanges;
}

FailureOr<TilingResult> MatMulOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  Value lhs = getLhs();
  Value rhs = getRhs();

  llvm::SmallVector<mlir::OpFoldResult, 4> strides(offsets.size(), builder.getIndexAttr(1));

  auto lhsSlice = builder.create<tensor::ExtractSliceOp>(
    loc, lhs.getType().cast<RankedTensorType>(), lhs, offsets, sizes, strides);

  auto rhsSlice = builder.create<tensor::ExtractSliceOp>(
    loc, rhs.getType().cast<RankedTensorType>(), rhs, offsets, sizes, strides);

  auto tiledMatmul = builder.create<mlir::mini::MatMulOp>(loc, lhsSlice, rhsSlice, getResult().getType());

  TilingResult result;
  result.tiledOps.push_back(tiledMatmul);
  result.tiledValues.push_back(tiledMatmul.getResult());

  return result;
}

LogicalResult MatMulOp::getResultTilePosition(OpBuilder &builder, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets, SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

//===----------------------------------------------------------------------===//
// AddOp: Tiling Interface Methods
//===----------------------------------------------------------------------===//

SmallVector<mlir::utils::IteratorType> AddOp::getLoopIteratorTypes() {
  return {mlir::utils::IteratorType::parallel, mlir::utils::IteratorType::parallel};
}

SmallVector<Range> AddOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  SmallVector<Range> loopRanges;

  Value lhs = getLhs();

  Value m = builder.create<tensor::DimOp>(loc, lhs, 0);
  Value n = builder.create<tensor::DimOp>(loc, lhs, 1);

  OpFoldResult lowerBound = builder.getIndexAttr(0);
  OpFoldResult step = builder.getIndexAttr(1);

  loopRanges.push_back(Range{lowerBound, m, step});
  loopRanges.push_back(Range{lowerBound, n, step});

  return loopRanges;
}

FailureOr<TilingResult> AddOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  Value lhs = getLhs();
  Value rhs = getRhs();

  auto strides = llvm::SmallVector<OpFoldResult, 4>(offsets.size(), builder.getIndexAttr(1));

  auto lhsSlice = builder.create<tensor::ExtractSliceOp>(
    loc, lhs.getType().cast<RankedTensorType>(), lhs, offsets, sizes, strides);

  auto rhsSlice = builder.create<tensor::ExtractSliceOp>(
    loc, rhs.getType().cast<RankedTensorType>(), rhs, offsets, sizes, strides);

  auto resultType = lhsSlice.getType();
  Value tiledAdd = builder.create<AddOp>(loc, resultType, lhsSlice, rhsSlice);

  TilingResult result;
  result.tiledOps.push_back(tiledAdd.getDefiningOp());
  result.tiledValues.push_back(tiledAdd);

  return result;
}

LogicalResult AddOp::getResultTilePosition(OpBuilder &builder, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets, SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

//===----------------------------------------------------------------------===//
// Bufferization Interface Methods for MatMulOp and AddOp
//===----------------------------------------------------------------------===//

namespace {

/// Bufferization support for MatMulOp
struct MatMulOpBufferization : public BufferizableOpInterface::ExternalModel<MatMulOpBufferization, MatMulOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const override {
    auto matMulOp = cast<MatMulOp>(op);
    FailureOr<Value> lhsBuffer = getBuffer(rewriter, matMulOp.getOperand(0), options);
    FailureOr<Value> rhsBuffer = getBuffer(rewriter, matMulOp.getOperand(1), options);
    FailureOr<Value> resultBuffer = getBuffer(rewriter, matMulOp.getResult(), options);

    if (failed(lhsBuffer) || failed(rhsBuffer) || failed(resultBuffer))
      return failure();

    rewriter.create<MatMulOp>(matMulOp.getLoc(), *lhsBuffer, *rhsBuffer, *resultBuffer);
    rewriter.replaceOp(matMulOp, *resultBuffer);
    return success();
  }
};

/// Bufferization support for AddOp
struct AddOpBufferization : public BufferizableOpInterface::ExternalModel<AddOpBufferization, AddOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const override {
    auto addOp = cast<AddOp>(op);
    FailureOr<Value> lhsBuffer = getBuffer(rewriter, addOp.getOperand(0), options);
    FailureOr<Value> rhsBuffer = getBuffer(rewriter, addOp.getOperand(1), options);
    FailureOr<Value> resultBuffer = getBuffer(rewriter, addOp.getResult(), options);

    if (failed(lhsBuffer) || failed(rhsBuffer) || failed(resultBuffer))
      return failure();

    rewriter.create<AddOp>(addOp.getLoc(), *lhsBuffer, *rhsBuffer, *resultBuffer);
    rewriter.replaceOp(addOp, *resultBuffer);
    return success();
  }
};

} // namespace

void mlir::mini::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension([](MLIRContext *ctx, MiniDialect *dialect) {
    MatMulOp::attachInterface<MatMulOpBufferization>(*ctx);
    AddOp::attachInterface<AddOpBufferization>(*ctx);
  });
}

