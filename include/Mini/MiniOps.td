//===- MiniOps.td - Mini Dialect Operation Definitions -----------*- tablegen -*-===//
//
// This file defines the operations of the Mini dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MINI_OPS
#define MINI_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ShapeInferenceInterface.td"
include "mlir/Interfaces/TilingInterface.td"

// Define the 'mini' dialect in the ODS framework.
def Mini_Dialect : Dialect {
  let name = "mini";
  let cppNamespace = "::mlir::mini";
}

// Base class for mini dialect operations.
class Mini_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mini_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Mini Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

def MatMulOp : Mini_Op<"matmul",[Pure, DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]> ]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication on two input tensors.

    Example:

    ```mlir
    %3 = mini.matmul %0, %1
        : (tensor<24x32xi8>, tensor<32x16xi8>) -> tensor<24x16xi8>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  // Define the assembly format.
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  // Builders for creating the operation.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Type":$resultType), [{
      build($_builder, $_state, resultType, lhs, rhs);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : Mini_Op<"add",[Pure, DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Element-wise addition operation";
  let description = [{
    Performs element-wise addition on two input tensors.

    Example:

    ```mlir
    %4 = mini.add %3, %2
        : (tensor<24x16xi8>, tensor<24x16xi8>) -> tensor<24x16xi8>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  // Define the assembly format.
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  // Builders for creating the operation.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Type":$resultType), [{
      build($_builder, $_state, resultType, lhs, rhs);
    }]>
  ];
}

#endif // MINI_OPS

