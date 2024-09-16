//===- mini-opt.cpp - Mini Dialect Optimizer --------------------*- C++ -*-===//
//
// This file implements a tool for testing the Mini dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "Mini/MiniDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "Mini/TileMiniOpsPass.h"
#include "Mini/Passes.h" 


int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::mini::registerTileMiniOpsPass(); 
  
  mlir::DialectRegistry registry;
  // Register the Func dialect
  registry.insert<mlir::func::FuncDialect>();
  // Register the Arith dialect
  registry.insert<mlir::arith::ArithDialect>();
  // Register the Mini dialect
  registry.insert<mlir::mini::MiniDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Mini optimizer driver\n", registry));
}

