#ifndef MINI_BUFFERIZABLE_OPS_H
#define MINI_BUFFERIZABLE_OPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

namespace mlir {
namespace mini {

/// Register bufferizable op interface external models for the Mini dialect.
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

/// Register bufferizable ops for the Mini dialect.
void registerBufferizableOps(); // Declare the function here

} // namespace mini
} // namespace mlir

#endif // MINI_BUFFERIZABLE_OPS_H

