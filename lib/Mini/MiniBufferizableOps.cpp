#include "Mini/MiniBufferizableOps.h"
#include "Mini/MiniOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::mini;

namespace {

/// This class provides bufferization for the MatMulOp.
struct MatMulOpBufferization : public BufferizableOpInterface::ExternalModel<MatMulOpBufferization, MatMulOp> {
  
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const override {
    auto matMulOp = cast<MatMulOp>(op);
    Location loc = matMulOp.getLoc();

    // Get buffers for operands
    Value lhsBuffer = getBuffer(rewriter, matMulOp.getOperand(0), options);
    if (!lhsBuffer)
      return failure();
    Value rhsBuffer = getBuffer(rewriter, matMulOp.getOperand(1), options);
    if (!rhsBuffer)
      return failure();

    // Get buffer for the result
    Value resultBuffer = getBuffer(rewriter, matMulOp.getResult(), options);
    if (!resultBuffer)
      return failure();

    // Create a new bufferized MatMul operation
    rewriter.create<MatMulOp>(loc, lhsBuffer, rhsBuffer, resultBuffer.getType());
    
    // The original MatMulOp is no longer needed
    rewriter.eraseOp(matMulOp);
    
    return success();
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand, const BufferizationOptions &options) const override {
    // MatMulOp writes to its output operand.
    return opOperand.getOperandNumber() == 2;  // Assuming the result is the third operand.
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const BufferizationOptions &options) const override {
    // MatMulOp reads from its input operands (lhs and rhs).
    return opOperand.getOperandNumber() < 2;
  }

  LogicalResult verifyAnalysis(Operation *op, const AnalysisState &state) const override {
    // Perform analysis verification if necessary.
    return success();
  }

  SmallVector<AliasingOpResult> getAliasingOpResults(Operation *op, OpOperand &opOperand, const AnalysisState &state) const override {
    // MatMulOp has a single result which aliases the third operand (result buffer).
    if (opOperand.getOperandNumber() == 2) {
      return {AliasingOpResult{op->getResult(0), BufferRelation::Equivalent}};
    }
    return {};
  }
};

/// This class provides bufferization for the AddOp.
struct AddOpBufferization : public BufferizableOpInterface::ExternalModel<AddOpBufferization, AddOp> {

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const override {
    auto addOp = cast<AddOp>(op);
    Location loc = addOp.getLoc();

    // Get buffers for operands
    Value lhsBuffer = getBuffer(rewriter, addOp.getOperand(0), options);
    if (!lhsBuffer)
      return failure();
    Value rhsBuffer = getBuffer(rewriter, addOp.getOperand(1), options);
    if (!rhsBuffer)
      return failure();

    // Get buffer for the result
    Value resultBuffer = getBuffer(rewriter, addOp.getResult(), options);
    if (!resultBuffer)
      return failure();

    // Create a new bufferized Add operation
    rewriter.create<AddOp>(loc, lhsBuffer, rhsBuffer, resultBuffer.getType());

    // The original AddOp is no longer needed
    rewriter.eraseOp(addOp);

    return success();
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand, const BufferizationOptions &options) const override {
    // AddOp writes to its output operand.
    return opOperand.getOperandNumber() == 2;  // Assuming the result is the third operand.
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const BufferizationOptions &options) const override {
    // AddOp reads from its input operands (lhs and rhs).
    return opOperand.getOperandNumber() < 2;
  }

  LogicalResult verifyAnalysis(Operation *op, const AnalysisState &state) const override {
    // Perform analysis verification if necessary.
    return success();
  }

  SmallVector<AliasingOpResult> getAliasingOpResults(Operation *op, OpOperand &opOperand, const AnalysisState &state) const override {
    // AddOp has a single result which aliases the third operand (result buffer).
    if (opOperand.getOperandNumber() == 2) {
      return {AliasingOpResult{op->getResult(0), BufferRelation::Equivalent}};
    }
    return {};
  }
};

/// Register bufferization interfaces.
void mlir::mini::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<MatMulOp, MatMulOpBufferization>();
  registry.addOpInterface<AddOp, AddOpBufferization>();
}

}  // namespace



