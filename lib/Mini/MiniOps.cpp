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
//#include "Mini/ShapeInferenceInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h" 
//#include "Mini/TilingInterface.h"

using namespace mlir;
using namespace mlir::mini;

#define GET_OP_CLASSES
#include "Mini/MiniOps.cpp.inc"


static int64_t getConstantIntValue(OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>())
    return attr.cast<IntegerAttr>().getInt();
  // Handle Value case if necessary
  llvm_unreachable("Expected constant integer value");
}

SmallVector<utils::IteratorType> MatMulOp::getLoopIteratorTypes() {
  // matmul is a 2D operation with loops over the M, N, and K dimensions
  // the outer loops (M, N) are parallel, and the inner loop (K) is a reduction.
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

  // Use tensor::DimOp for extracting dimensions.
  //Value m = builder.create<tensor::DimOp>(loc, lhs, builder.getIndexAttr(0));
  Value m = builder.create<tensor::DimOp>(loc, lhs, 0);
  //Value n = builder.create<tensor::DimOp>(loc, rhs, builder.getIndexAttr(1));
  Value n = builder.create<tensor::DimOp>(loc, rhs, 1);
  //Value k = builder.create<tensor::DimOp>(loc, lhs, builder.getIndexAttr(1));
  Value k = builder.create<tensor::DimOp>(loc, lhs, 1);
  
  // Create OpFoldResults for bounds and step.
  OpFoldResult lowerBound = builder.getIndexAttr(0); // lower bound 0
  OpFoldResult step = builder.getIndexAttr(1);       // step 1

  // Convert upper bounds (m, n, k) to OpFoldResult.
  OpFoldResult upperBoundM = m;
  OpFoldResult upperBoundN = n;
  OpFoldResult upperBoundK = k;

  // Create ranges for the loops: Rows (m), Columns (n), and Reduction (k).
  loopRanges.push_back(Range{lowerBound, upperBoundM, step});
  loopRanges.push_back(Range{lowerBound, upperBoundN, step});
  loopRanges.push_back(Range{lowerBound, upperBoundK, step});

  return loopRanges;
}




FailureOr<TilingResult> MatMulOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  Value lhs = getLhs();
  Value rhs = getRhs();


  // Create a slice of the lhs and rhs tensors using the offsets and sizes.
  llvm::SmallVector<mlir::OpFoldResult, 4> strides(offsets.size(), builder.getIndexAttr(1)); // Default stride of 1

  auto lhsSlice = builder.create<tensor::ExtractSliceOp>(
    loc, lhs.getType().cast<RankedTensorType>(), lhs, offsets, sizes, strides, /* attrs= */ ArrayRef<NamedAttribute>{});

  auto rhsSlice = builder.create<tensor::ExtractSliceOp>(
    loc, rhs.getType().cast<RankedTensorType>(), rhs, offsets, sizes, strides, /* attrs= */ ArrayRef<NamedAttribute>{});

  // Create the tiled matmul operation using the slices.
  auto tiledMatmul = builder.create<mlir::mini::MatMulOp>(loc, lhsSlice, rhsSlice, getResult().getType());

  // Return the tiled operation as the result.
  TilingResult result;
  result.tiledOps.push_back(tiledMatmul);
  result.tiledValues.push_back(tiledMatmul.getResult());

  return result;
}

LogicalResult MatMulOp::getResultTilePosition(OpBuilder &builder, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets, SmallVector<OpFoldResult> &resultSizes) {

  // For now, simply pass through the offsets and sizes.
  // In matrix multiplication, the result tile position is directly related to the input tile.
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());

  return success();
}

SmallVector<mlir::utils::IteratorType> AddOp::getLoopIteratorTypes() {
  // Both dimensions of the addition are parallel loops.
  return {mlir::utils::IteratorType::parallel, 
          mlir::utils::IteratorType::parallel};
}


SmallVector<Range> AddOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  SmallVector<Range> loopRanges;

  Value lhs = getLhs();

  // Use tensor::DimOp for extracting dimensions.
  //Value m = builder.create<tensor::DimOp>(loc, lhs, builder.getIndexAttr(0));
  Value m = builder.create<tensor::DimOp>(loc, lhs, 0);
  //Value n = builder.create<tensor::DimOp>(loc, lhs, builder.getIndexAttr(1));
  Value n = builder.create<tensor::DimOp>(loc, lhs, 1);

  // Convert the bounds (0, m, n) to OpFoldResult.
  OpFoldResult lowerBound = builder.getIndexAttr(0); // lower bound 0
  OpFoldResult step = builder.getIndexAttr(1);       // step 1

  OpFoldResult upperBoundM = m;
  OpFoldResult upperBoundN = n;

  // Add ranges for each of the two loops: M and N.
  loopRanges.push_back(Range{lowerBound, upperBoundM, step});
  loopRanges.push_back(Range{lowerBound, upperBoundN, step});

  return loopRanges;
}


llvm::FailureOr<TilingResult> AddOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {

  Location loc = getLoc();
  Value lhs = getLhs();
  Value rhs = getRhs();

  // Create slices for the LHS and RHS tensors
  //auto lhsSlice = builder.create<tensor::ExtractSliceOp>(loc, lhs, offsets, sizes);
  //auto rhsSlice = builder.create<tensor::ExtractSliceOp>(loc, rhs, offsets, sizes);

  auto strides = llvm::SmallVector<OpFoldResult, 4>(offsets.size(), builder.getIndexAttr(1));  // Default stride of 1 for each dimension

  auto lhsSlice = builder.create<tensor::ExtractSliceOp>(
    loc, lhs.getType().cast<RankedTensorType>(), lhs, offsets, sizes, strides, /*attrs=*/ArrayRef<NamedAttribute>{});

  auto rhsSlice = builder.create<tensor::ExtractSliceOp>(
    loc, rhs.getType().cast<RankedTensorType>(), rhs, offsets, sizes, strides, /*attrs=*/ArrayRef<NamedAttribute>{});



  // Create the tiled AddOp with the slices
  auto resultType = lhsSlice.getType();  // The result type is the same as the slice type
  Value tiledAdd = builder.create<AddOp>(loc, resultType, lhsSlice, rhsSlice);

  // Return the tiled result in a TilingResult struct
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

