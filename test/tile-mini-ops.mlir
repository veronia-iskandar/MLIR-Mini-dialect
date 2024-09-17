// Test the tile-mini-ops pass
func.func @test(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = mini.matmul %arg0, %arg1 : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
  %1 = mini.add %0, %arg0 : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
}



