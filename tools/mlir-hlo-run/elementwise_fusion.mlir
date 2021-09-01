module {
  func @main(%arg0: tensor<4x16xi32>, %arg1: tensor<4x16xi32>) -> tensor<2x4xi32> {
    %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4x16xi32>, tensor<4x16xi32>) -> tensor<4x16xi32>
    %1 = "mhlo.subtract"(%0, %arg0) : (tensor<4x16xi32>, tensor<4x16xi32>) -> tensor<4x16xi32>
    %2 = "mhlo.slice"(%1) {limit_indices = dense<[2, 8]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x16xi32>) -> tensor<2x8xi32>
    %3 = "mhlo.multiply"(%0, %1) : (tensor<4x16xi32>, tensor<4x16xi32>) -> tensor<4x16xi32>
    %4 = "mhlo.slice"(%3) {limit_indices = dense<[2, 8]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x16xi32>) -> tensor<2x4xi32>
    return %4 : tensor<2x4xi32>
  }
}