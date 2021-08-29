/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/register_passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir-hlo/Transforms/register_passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Support/MlirOptMain.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

size_t sizeOfRankedTensorDesc(mlir::RankedTensorType type) {
  size_t s = sizeof(void *) * 2;
  s += sizeof(int64_t) * (type.getShape().size() * 2 + 1);
  return s;
}

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::disc_ral::registerAllDiscRalPasses();
  mlir::hlo::registerAllHloPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::chlo::HloClientDialect>();
  registry.insert<mlir::lmhlo::LmhloDialect>();
  registry.insert<mlir::lmhlo_gpu::LmhloGpuDialect>();
  registry.insert<mlir::disc_ral::RalDialect>();

  mlir::MLIRContext mlir_ctx(registry);

  std::ifstream fs(argv[1]);
  std::string content = std::string(std::istreambuf_iterator<char>(fs),
                                    std::istreambuf_iterator<char>());

  auto module = mlir::parseSourceString(content, &mlir_ctx);

  auto entry_function = module->lookupSymbol<mlir::FuncOp>("main");
  auto entry_func_type = entry_function.getType();

  mlir::PassManager pm(&mlir_ctx);
  mlir_ctx.disableMultithreading();
  pm.enableIRPrinting();

  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeToMemrefPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createTensorBufferizePass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createStdBufferizePass());
  pm.addPass(mlir::mhlo::createLegalizeToLhloPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createBufferHoistingPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createBufferDeallocationPass());

  pm.addNestedPass<mlir::FuncOp>(mlir::lmhlo::createLegalizeLhloToLinalgPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createConvertLinalgToLoopsPass());
  // pm.addPass(mlir::createConvertLinalgToLLVMPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());

  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createMemRefToLLVMPass());
  pm.addPass(mlir::createLowerToLLVMPass());

  pm.run(*module);

  mlir::registerLLVMDialectTranslation(*module->getContext());
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/true ? 0 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
 

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      *module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // engine->dumpToObjectFile("/home/lipracer/work/mlir-hlo/build/add_bin.txt");

  auto mainFunction = engine->lookup("main");
  assert(mainFunction && "failed to get main function");

  const size_t data_size = 6;
  std::vector<float> vec_data(data_size, 0.0);
  std::iota(vec_data.begin(), vec_data.end(), 1.0);
  float* raw_data = vec_data.data();

  struct dyn_descriptor {
    void *allocated;
    void *aligned;
    int64_t offset;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
  };

  auto makeDynDesc = [](void *data,
                        mlir::RankedTensorType type) -> dyn_descriptor {
    dyn_descriptor desc;
    desc.allocated = data;
    desc.aligned = data;
    desc.offset = 0;
    desc.sizes.resize(type.getShape().size());
    desc.strides.resize(type.getShape().size());
    std::copy(type.getShape().begin(), type.getShape().end(),
              desc.sizes.begin());
    for (size_t i = 0; i < type.getShape().size(); ++i) {
      desc.strides[i] = std::accumulate(
          desc.sizes.begin(), desc.sizes.begin() + i, 1, std::multiplies<>());
    }
    return desc;
  };

  std::vector<void *> arguments;
  arguments.reserve(16);
  for(auto input_ty : entry_func_type.getInputs()) {
    auto desc = makeDynDesc(raw_data, input_ty.cast<mlir::RankedTensorType>());
    arguments.push_back(&desc.allocated);
    arguments.push_back(&desc.aligned);
    arguments.push_back(&desc.offset);
    for (size_t i = 0; i < desc.sizes.size(); ++i)
      arguments.push_back(&desc.sizes[i]);
    for (size_t i = 0; i < desc.strides.size(); ++i)
      arguments.push_back(&desc.strides[i]);
  }

  std::vector<char> result(sizeOfRankedTensorDesc(
      entry_func_type.getResult(0).cast<mlir::RankedTensorType>()));
  arguments.push_back(&result.front());

  auto main_func_ptr = mainFunction.get();
  main_func_ptr(arguments.data());

  auto result_desc =
      reinterpret_cast<MemRefDescriptor<float, 2> *>(&result.front());

  std::cout << "sizes:" << result_desc->sizes[0] << " " << result_desc->sizes[1]
            << " strides:";
  std::cout << result_desc->strides[0] << " " << result_desc->strides[1]
            << std::endl;

  std::for_each(result_desc->allocated, result_desc->allocated + 6,
                [](auto it) { std::cout << it << " "; });
  std::cout << std::endl;

  return 0;
}
