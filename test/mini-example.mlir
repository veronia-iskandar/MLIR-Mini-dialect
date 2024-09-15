func.func @example() -> tensor<24x16xi8> {
  %0 = arith.constant dense<1> : tensor<24x32xi8>
  %1 = arith.constant dense<1> : tensor<32x16xi8>
  %2 = arith.constant dense<1> : tensor<24x16xi8>
  %3 = mini.matmul %0, %1 : tensor<24x32xi8>, tensor<32x16xi8> -> tensor<24x16xi8>
  %4 = mini.add %3, %2 : tensor<24x16xi8>, tensor<24x16xi8> -> tensor<24x16xi8>
  return %4 : tensor<24x16xi8>
}

