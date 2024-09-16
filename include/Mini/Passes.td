//===- Passes.td - Mini Dialect Passes --------------------------*- tablegen -*-===//
//
// This file defines the passes for the Mini dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def TileMiniOpsPass : Pass<"tile-mini-ops", "func::FuncOp"> {
  let summary = "Tile mini.matmul and mini.add operations to 8x8 tiles.";
  let description = [{
    This pass applies tiling to the mini.matmul and mini.add operations,
    using a tile size of 8x8.
  }];
}

