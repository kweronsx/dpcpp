// COMPILE: build/bin/clang++ -DSYCL_EXT_ONEAPI_KERNEL_COMPILER=1 sdg/02.cpp -o sdg/02.out -lze_loader

// REQUIRES SPIRV module.cpp, which can be built like so:
// build/bin/clang++ -fsycl -fsycl-device-only -fno-sycl-instrument-device-code sdg/module.cpp -o sdg/dg.bc
// build/bin/llvm-spirv -spirv-ext=+SPV_INTEL_global_variable_decorations sdg/dg.bc -o sdg/dg.spv

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <level_zero/ze_api.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    ze_result_t _st = (call);                                                  \
    if (_st != ZE_RESULT_SUCCESS) {                                            \
      std::cerr << #call " failed: 0x" << std::hex << _st << std::dec << "\n"; \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static std::vector<uint8_t> readFile(const char *Path) {
  std::ifstream F(Path, std::ios::binary | std::ios::ate);
  if (!F) {
    std::cerr << "cannot open SPIR-V file: " << Path << "\n";
    std::exit(1);
  }
  size_t Size = static_cast<size_t>(F.tellg());
  F.seekg(0);
  std::vector<uint8_t> Data(Size);
  F.read(reinterpret_cast<char *>(Data.data()), Size);
  return Data;
}

int test_device_global(int argc, char **argv) {
  const char *SpvPath = argc > 1 ? argv[1] : "dg.spv";
  std::vector<uint8_t> Spirv = readFile(SpvPath);

  // 1. Initialize and pick the first GPU driver + device.
  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  uint32_t DriverCount = 0;
  CHECK(zeDriverGet(&DriverCount, nullptr));
  assert(DriverCount > 0 && "no Level Zero drivers");
  std::vector<ze_driver_handle_t> Drivers(DriverCount);
  CHECK(zeDriverGet(&DriverCount, Drivers.data()));
  ze_driver_handle_t Driver = Drivers[0];

  uint32_t DevCount = 0;
  CHECK(zeDeviceGet(Driver, &DevCount, nullptr));
  assert(DevCount > 0 && "no devices for driver 0");
  std::vector<ze_device_handle_t> Devices(DevCount);
  CHECK(zeDeviceGet(Driver, &DevCount, Devices.data()));
  ze_device_handle_t Device = Devices[0];

  // 2. Context.
  ze_context_desc_t CtxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t Context;
  CHECK(zeContextCreate(Driver, &CtxDesc, &Context));

  // 3. Build the module from SPIR-V.
  //    "-ze-take-global-address" tells IGC to expose module-scope globals via
  //    zeModuleGetGlobalPointer. It is enabled by default on newer IGC, but
  //    passing it explicitly is harmless and works on older stacks too.
  ze_module_desc_t ModDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
  ModDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  ModDesc.inputSize = Spirv.size();
  ModDesc.pInputModule = Spirv.data();
  ModDesc.pBuildFlags = "-ze-take-global-address";
  ModDesc.pConstants = nullptr;

  ze_module_handle_t Module;
  ze_module_build_log_handle_t BuildLog;
  ze_result_t BuildSt =
      zeModuleCreate(Context, Device, &ModDesc, &Module, &BuildLog);
  if (BuildSt != ZE_RESULT_SUCCESS) {
    size_t LogSize = 0;
    zeModuleBuildLogGetString(BuildLog, &LogSize, nullptr);
    std::vector<char> Log(LogSize);
    zeModuleBuildLogGetString(BuildLog, &LogSize, Log.data());
    std::cerr << "module build failed:\n" << Log.data() << "\n";
    std::exit(1);
  }
  zeModuleBuildLogDestroy(BuildLog);

  // 4. Create the kernel. The `extern "C"` free function `ff_dg_adder` is
  //    emitted with the `__sycl_kernel_` prefix as its SPIR-V entry point.
  ze_kernel_desc_t KernDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
  KernDesc.pKernelName = "__sycl_kernel_ff_dg_adder";

  ze_kernel_handle_t Kernel, KernelSwap;
  CHECK(zeKernelCreate(Module, &KernDesc, &Kernel));

  KernDesc.pKernelName = "__sycl_kernel_ff_swap";
  CHECK(zeKernelCreate(Module, &KernDesc, &KernelSwap));

  // 5. Immediate, synchronous command list (each append blocks to completion).
  ze_command_queue_desc_t CqDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  CqDesc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  ze_command_list_handle_t CmdList;
  CHECK(zeCommandListCreateImmediate(Context, Device, &CqDesc, &CmdList));

  // 6. Get the device global address. The Level Zero name is the
  //    sycl-unique-id (Itanium-mangled variable name), i.e. "_Z2DG" for
  //    `device_global<int32_t> DG;`.
  size_t DgSize = 0;
  void *DgAddrStruct = nullptr;
  CHECK(zeModuleGetGlobalPointer(Module, "_Z2DG", &DgSize, &DgAddrStruct));
  // `DG` has no `device_image_scope`, so `_Z2DG` is a `device_global` wrapper
  // { int32_t *usmptr; int32_t init_val; } (16 bytes), not the value itself.
  // The value lives in a separate allocation referenced by `usmptr` (offset 0).
  // The SYCL runtime normally allocates that and patches `usmptr`; with pure
  // Level Zero we do it ourselves.
  assert(DgSize == sizeof(void *) + sizeof(int32_t) + /*padding=*/4);

  ze_device_mem_alloc_desc_t DevAllocDesc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void *DgAddr = nullptr;
  CHECK(zeMemAllocDevice(Context, &DevAllocDesc, sizeof(int32_t),
                         alignof(int32_t), Device, &DgAddr));
  // Write the value-buffer address into `usmptr` at offset 0 of the wrapper.
  CHECK(zeCommandListAppendMemoryCopy(CmdList, DgAddrStruct, &DgAddr,
                                      sizeof(DgAddr), nullptr, 0, nullptr));
  CHECK(zeCommandListHostSynchronize(CmdList, UINT64_MAX));

  size_t DgDisSize = 0;
  void *DgDisAddr = nullptr;
  CHECK(zeModuleGetGlobalPointer(Module, "_Z6DG_DIS", &DgDisSize, &DgDisAddr));
  assert(DgDisSize == 8);

  auto checkGlob = [=](int32_t expected) {
    int32_t val = -1;

    CHECK(zeCommandListAppendMemoryCopy(CmdList, &val, DgAddr, sizeof(val),
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListHostSynchronize(CmdList, UINT64_MAX));

    std::cout << "val: " << val << " == " << expected << std::endl;
    assert(val == expected);
  };

  auto setGlob = [CmdList, DgAddr](int32_t value) {
    CHECK(zeCommandListAppendMemoryCopy(CmdList, DgAddr, &value, sizeof(value),
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListHostSynchronize(CmdList, UINT64_MAX));
  };

  auto runAdder = [=](int32_t value) {
    CHECK(zeKernelSetArgumentValue(Kernel, 0, sizeof(value), &value));
    CHECK(zeKernelSetGroupSize(Kernel, 1, 1, 1));
    ze_group_count_t Groups = {1, 1, 1};
    CHECK(zeCommandListAppendLaunchKernel(CmdList, Kernel, &Groups, nullptr, 0,
                                          nullptr));
    CHECK(zeCommandListHostSynchronize(CmdList, UINT64_MAX));
  };

  setGlob(0);
  checkGlob(0);

  // Set the DG.
  setGlob(123);
  checkGlob(123);

  // Run a kernel using it.
  setGlob(-17);
  runAdder(123);
  checkGlob(123 - 17);

  // Test global with `device_image_scope`. The `ff_swap` kernel takes
  // an `int64_t *`, so the argument must be a device pointer: the kernel writes
  // through it, and the swapped value persists and is visible on the host once
  // we copy it back.
  int64_t *valBuf = nullptr;
  CHECK(zeMemAllocDevice(Context, &DevAllocDesc, sizeof(int64_t),
                         alignof(int64_t), Device,
                         reinterpret_cast<void **>(&valBuf)));

  int64_t hostVal = -1;
  CHECK(zeCommandListAppendMemoryCopy(CmdList, valBuf, &hostVal,
                                      sizeof(hostVal), nullptr, 0, nullptr));
  CHECK(zeCommandListHostSynchronize(CmdList, UINT64_MAX));

  // Seed the `device_image_scope` global with zero before the first swap so the
  // swapped-out value is well-defined.
  int64_t dgDisInit = 0;
  CHECK(zeCommandListAppendMemoryCopy(CmdList, DgDisAddr, &dgDisInit,
                                      sizeof(dgDisInit), nullptr, 0, nullptr));
  CHECK(zeCommandListHostSynchronize(CmdList, UINT64_MAX));

  auto runSwap = [&]() {
    CHECK(zeKernelSetArgumentValue(KernelSwap, 0, sizeof(valBuf), &valBuf));
    CHECK(zeKernelSetGroupSize(KernelSwap, 1, 1, 1));
    ze_group_count_t Groups = {1, 1, 1};
    CHECK(zeCommandListAppendLaunchKernel(CmdList, KernelSwap, &Groups, nullptr,
                                          0, nullptr));
    CHECK(zeCommandListHostSynchronize(CmdList, UINT64_MAX));
  };

  auto readValBuf = [&]() {
    int64_t out = 0;
    CHECK(zeCommandListAppendMemoryCopy(CmdList, &out, valBuf, sizeof(out),
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListHostSynchronize(CmdList, UINT64_MAX));
    return out;
  };

  runSwap();
  assert(readValBuf() == 0);
  runSwap();
  assert(readValBuf() == -1);

  CHECK(zeMemFree(Context, valBuf));
  CHECK(zeMemFree(Context, DgAddr));
  zeKernelDestroy(KernelSwap);
  zeKernelDestroy(Kernel);
  zeModuleDestroy(Module);
  zeCommandListDestroy(CmdList);
  zeContextDestroy(Context);
  return 0;
}

#ifndef MCR_TEST_COUNT
#define MCR_TEST_COUNT 5
#endif

int main(int argc, char **argv) {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  auto cwd = std::filesystem::path(argv[0]).remove_filename();
  auto canonical = std::filesystem::canonical(absolute(cwd));
  std::filesystem::current_path(canonical);

  constexpr std::size_t testCount{MCR_TEST_COUNT};
  std::size_t testIteration{1}, failed{};
  int constexpr OK = 0;

  for (; testIteration <= testCount; ++testIteration) {
    std::cout << "Test iteration: " << testIteration << " / " << testCount;
    std::cout << std::endl;

    if (test_device_global(argc, argv) != OK) {
      ++failed;
      break;
    }
  }

  return failed;
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
