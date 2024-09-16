//===- MiniDialect.td - Mini Dialect Definitions ----------------*- tablegen -*-===//
//
// This file defines the Mini dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/Dialect.td"

// Define the Mini dialect in the ODS framework.
def Mini_Dialect : Dialect {
  let name = "mini";
  let cppNamespace = "::mlir::mini";
}

