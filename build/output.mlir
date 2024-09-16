module {
  func.func @example() -> tensor<24x16xi8> {
    %cst = arith.constant dense<1> : tensor<24x32xi8>
    %cst_0 = arith.constant dense<1> : tensor<32x16xi8>
    %cst_1 = arith.constant dense<1> : tensor<24x16xi8>
    %0 = mini.matmul %cst, %cst_0 : tensor<24x32xi8>, tensor<32x16xi8> -> tensor<24x16xi8>
    %1 = mini.add %0, %cst_1 : tensor<24x16xi8>, tensor<24x16xi8> -> tensor<24x16xi8>
    return %1 : tensor<24x16xi8>
  }
}

