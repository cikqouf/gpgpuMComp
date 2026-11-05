#include <fstream>
#include <string.h>

#include "engineBase.hpp"
#include "matHppOnlyLib/mat.hpp"

namespace ComputeModule {

static std::vector<char> shaderReader(const char *path) {
  std::ifstream file(path, std::ios_base::ate | std::ios_base::binary);

  size_t size = file.tellg();
  file.seekg(0);
  std::vector<char> code(size);
  file.read(code.data(), size);
  return code;
}

class compComponent {

public:
  EngineBase::initBase initCore;

public:
  compComponent() : initCore(vk::PhysicalDeviceType::eDiscreteGpu) {}
};

enum class gMatFlagBits : uint32_t {
  eNone = 0u, // buffer handle bind to no handle, default
  eCpu = 1u,  // buffer handle bind to cpuMappedMem, ususlly iGpu Mem
  eGpu = 2u   // buffer handle bind to gpuMem
};

template <typename T> struct gMat {

  friend class MemoryManger;

public:
  mat<T> m;

private:
  vk::raii::Buffer handle = nullptr;
  vk::raii::Buffer gpuHandle = nullptr;
  vk::DeviceSize offset = 0;
  vk::DeviceSize gpuOffset = 0;

public:
  gMatFlagBits flag = gMatFlagBits::eNone;

public:
  gMat() = delete;

  gMat(const vk::raii::Device &device, const mat<T> &inM,
       gMatFlagBits inF = gMatFlagBits::eNone) = delete;

  gMat(const vk::raii::Device &device, mat<T> &&inM,
       gMatFlagBits inF = gMatFlagBits::eNone)
      : m(inM),
        handle(vk::raii::Buffer(
            device,
            vk::BufferCreateInfo({}, inM.len * sizeof(T),
                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                     vk::BufferUsageFlagBits::eTransferSrc |
                                     vk::BufferUsageFlagBits::eTransferDst))),
        flag(inF) {}

public:
  // just a super simple impletation of A = a * A;
  // no fancy algorithm, only vulkan sync
  static void axA(const vk::raii::CommandBuffer &cmd,
                  const EngineBase::initBase &core, T a, gMat<T> &gBuf) {

    if (gBuf.flag == gMatFlagBits::eNone) {

      std::cerr << "axA(): gBuf is not bound to any Mem yet\n";
      throw std::runtime_error("");
    }

    if (gBuf.flag == gMatFlagBits::eCpu) {
      std::cerr << "axA(): send gBuf to gpu before do axA\n";
      throw std::runtime_error("");
    }

    auto queue = core.device.getQueue(core.queueFamilyIndex, 0u);

    std::vector<vk::DescriptorSetLayoutBinding> setBindings = {
        vk::DescriptorSetLayoutBinding(0u, vk::DescriptorType::eStorageBuffer,
                                       1u, vk::ShaderStageFlagBits::eCompute)

    };

    std::vector<vk::DescriptorPoolSize> poolSizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1u)};

    auto descSetPool =
        core.device.createDescriptorPool(vk::DescriptorPoolCreateInfo(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 32u,
            poolSizes.size(), poolSizes.data()));

    auto descSetLayouts =
        core.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(
            {}, setBindings.size(), setBindings.data()));

    auto descSets = core.device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo(descSetPool, 1u, &*descSetLayouts));

    const vk::DescriptorBufferInfo bufferInfos[] = {
        vk::DescriptorBufferInfo(gBuf.gpuHandle, 0, gBuf.m.len * sizeof(T))

    };

    core.device.updateDescriptorSets(
        {vk::WriteDescriptorSet(descSets[0], 0u, 0u, (uint32_t)poolSizes.size(),
                                vk::DescriptorType::eStorageBuffer, nullptr,
                                bufferInfos, nullptr, nullptr)},
        {});

    vk::PushConstantRange pushCs[] = {vk::PushConstantRange(
        vk::ShaderStageFlagBits::eCompute, 0u, sizeof(T))};

    auto pipeLayout = core.device.createPipelineLayout(
        vk::PipelineLayoutCreateInfo({}, 1, &*descSetLayouts, 1u, pushCs));

    auto shaderCode = ComputeModule::shaderReader("shaders/bin/comp.spv");
    auto shaderM = core.device.createShaderModule(vk::ShaderModuleCreateInfo(
        {}, shaderCode.size(), (uint32_t *)shaderCode.data()));
    auto compPipe = core.device.createComputePipeline(
        nullptr,
        vk::ComputePipelineCreateInfo(
            {},
            vk::PipelineShaderStageCreateInfo(
                {}, vk::ShaderStageFlagBits::eCompute, shaderM, "main"),
            pipeLayout));

    cmd.begin(vk::CommandBufferBeginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::DependencyFlagBits::eByRegion, {},
                        {

                            vk::BufferMemoryBarrier(
                                vk::AccessFlagBits::eTransferRead |
                                    vk::AccessFlagBits::eTransferWrite,
                                vk::AccessFlagBits::eShaderRead |
                                    vk::AccessFlagBits::eShaderWrite,
                                core.queueFamilyIndex, core.queueFamilyIndex,
                                gBuf.handle, 0, gBuf.m.len * sizeof(T)),

                            vk::BufferMemoryBarrier(
                                vk::AccessFlagBits::eTransferRead |
                                    vk::AccessFlagBits::eTransferWrite,
                                vk::AccessFlagBits::eShaderRead |
                                    vk::AccessFlagBits::eShaderWrite,
                                core.queueFamilyIndex, core.queueFamilyIndex,
                                gBuf.gpuHandle, 0, gBuf.m.len * sizeof(T))},
                        {});

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeLayout, 0u,
                           {descSets[0]}, {});

    cmd.pushConstants(pipeLayout, vk::ShaderStageFlagBits::eCompute, 0u,
                      vk::ArrayProxy<const T>{a});
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compPipe);

    cmd.dispatch(gBuf.m.len / 64 + 1, 1, 1);

    cmd.end();

    queue.submit({vk::SubmitInfo(0, nullptr, nullptr, 1, &*cmd, 0, nullptr)});

    // todo: 权宜之计...
    queue.waitIdle();
  }
};

class MemoryManger {
private:
  vk::DeviceSize memSize = 0;
  vk::raii::DeviceMemory cpuMem = nullptr;
  char *cpuMemMapped = nullptr;
  vk::raii::DeviceMemory gpuMem = nullptr;

  vk::DeviceSize cpuMemOffset = 0;
  vk::DeviceSize gpuMemOffset = 0;

public:
  MemoryManger() = delete;

  MemoryManger(const vk::raii::PhysicalDevice &physicalDevice,
               const vk::raii::Device &device, vk::DeviceSize offset,
               vk::DeviceSize inMemSize)
      : memSize(inMemSize),
        cpuMem(device.allocateMemory(vk::MemoryAllocateInfo(
            memSize, EngineBase::memTypeChoose(
                         physicalDevice,
                         vk::MemoryPropertyFlagBits::eHostCoherent |
                             vk::MemoryPropertyFlagBits::eHostVisible)))),
        cpuMemMapped((char *)cpuMem.mapMemory(offset, memSize)),
        gpuMem(device.allocateMemory(vk::MemoryAllocateInfo(
            memSize,
            EngineBase::memTypeChoose(
                physicalDevice, vk::MemoryPropertyFlagBits::eDeviceLocal)))) {

    std::cerr << "memManger: allocated memSize: " << memSize << " Bytes\n";
  }

  // 关于内存空泡问题, 本项目不管这个, 但是确实存在问题
  // 在 same gBuf do toGpu -> backToCpu -> toGpu 会因为 memMan 只维护
  // 一个全局的 offset 导致其实可以覆盖的内存没有去使用
  template <typename T>
  void toGpu(const vk::raii::CommandBuffer &cmd,
             const EngineBase::initBase &core, vk::DeviceSize dataSize,
             gMat<T> &gBuf) {
    if (gBuf.flag == gMatFlagBits::eCpu) {
      gBuf.flag = gMatFlagBits::eGpu;
    } else {
      if (gBuf.flag == gMatFlagBits::eNone) {
        std::cerr << "gBuf.handle " << *gBuf.handle
                  << " is not bound to cpuMem\n";
        throw std::runtime_error("");
      } else {
        std::cerr << "gBuf.gpuHandle " << *gBuf.gpuHandle
                  << " is bound to gpuMem already\n";
        throw std::runtime_error("");
      }
    }

    gBuf.gpuHandle = vk::raii::Buffer(
        core.device,
        vk::BufferCreateInfo({}, dataSize,
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferSrc |
                                 vk::BufferUsageFlagBits::eTransferDst));

    const auto req = gBuf.gpuHandle.getMemoryRequirements();
    const auto &ali = req.alignment;
    const auto &addSize = req.size;

    gpuMemOffset = gpuMemOffset % ali == 0
                       ? gpuMemOffset
                       : gpuMemOffset + ali - gpuMemOffset % ali;

    gBuf.gpuHandle.bindMemory(gpuMem, gpuMemOffset);
    gBuf.gpuOffset = gpuMemOffset;

    gpuMemOffset += addSize;

    // transfer
    auto queue = core.device.getQueue(core.queueFamilyIndex, 0u);

    cmd.begin(vk::CommandBufferBeginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    cmd.copyBuffer(gBuf.handle, gBuf.gpuHandle,
                   {vk::BufferCopy(0, 0, dataSize)

                   });

    cmd.end();
    queue.submit({vk::SubmitInfo(0, nullptr, nullptr, 1, &*cmd, 0, nullptr)});

    // queue.waitIdle();
  }

  template <typename T>
  void backToCpu(const vk::raii::CommandBuffer &cmd,
                 const EngineBase::initBase &core, vk::DeviceSize dataSize,
                 gMat<T> &gBuf) {
    if (gBuf.flag != gMatFlagBits::eGpu) {
      std::cerr << "gBuf is not on Gpu now\n";
      throw std::runtime_error("");
    } else {
      if (gBuf.flag == gMatFlagBits::eNone) {
        std::cerr << "gBuf is not even bound to cpu\n";
        throw std::runtime_error("");
      }
      gBuf.flag = gMatFlagBits::eCpu;
    }
    // transfer
    auto queue = core.device.getQueue(core.queueFamilyIndex, 0u);
    auto finished = core.device.createFence(vk::FenceCreateInfo());

    cmd.begin(vk::CommandBufferBeginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eTransfer,
                        vk::DependencyFlagBits::eByRegion, {},
                        {

                            vk::BufferMemoryBarrier(
                                vk::AccessFlagBits::eShaderRead |
                                    vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eTransferRead |
                                    vk::AccessFlagBits::eTransferWrite,

                                core.queueFamilyIndex, core.queueFamilyIndex,
                                gBuf.handle, 0, dataSize),

                            vk::BufferMemoryBarrier(
                                vk::AccessFlagBits::eShaderRead |
                                    vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eTransferRead |
                                    vk::AccessFlagBits::eTransferWrite,
                                core.queueFamilyIndex, core.queueFamilyIndex,
                                gBuf.gpuHandle, 0, dataSize)},
                        {});

    cmd.copyBuffer(gBuf.gpuHandle, gBuf.handle,
                   {vk::BufferCopy(0, 0, dataSize)

                   });

    cmd.end();
    queue.submit({vk::SubmitInfo(0, nullptr, nullptr, 1, &*cmd, 0, nullptr)},
                 finished);

    if (core.device.waitForFences({finished}, vk::True, UINT64_MAX) ==
        vk::Result::eSuccess) {
      core.device.resetFences({finished});

      memcpy((char *)gBuf.m.pData, (char *)cpuMemMapped + gBuf.offset,
             dataSize);
    }
  }

  template <typename T>
  auto &add(char *data, vk::DeviceSize dataSize, gMat<T> &gBuf) {

    if (gBuf.flag == gMatFlagBits::eNone) {
      gBuf.flag = gMatFlagBits::eCpu;
    } else {
      std::cerr << "gBuf has been added already\n";
      throw std::runtime_error("");
    }
    const auto req = gBuf.handle.getMemoryRequirements();
    const auto &ali = req.alignment;
    const auto &addSize = req.size;

    cpuMemOffset = cpuMemOffset % ali == 0
                       ? cpuMemOffset
                       : cpuMemOffset + ali - cpuMemOffset % ali;

    gBuf.handle.bindMemory(cpuMem, cpuMemOffset);
    gBuf.offset = cpuMemOffset;
    memcpy(cpuMemMapped + cpuMemOffset, data, dataSize);

    cpuMemOffset += addSize;

    return *this;
  }

  void unMap() { cpuMem.unmapMemory(); }
};

} // namespace ComputeModule
int main() {

  ComputeModule::compComponent compt;

  auto &core = compt.initCore;

  /* SSBO mem */
  vk::DeviceSize constexpr memSize = 1024'000;
  /* memory Mannager */
  ComputeModule::MemoryManger memMan(core.physicalDevice, core.device, 0,
                                     memSize);
  // process
  ComputeModule::gMat<float> gM1(core.device, mat<float>(24, 17, 1.f));

  memMan.add((char *)gM1.m.pData, gM1.m.len * sizeof(float), gM1);
  // do some computation
  auto cmdPool = core.device.createCommandPool(
      vk::CommandPoolCreateInfo({}, core.queueFamilyIndex));
  auto cmds = core.device.allocateCommandBuffers(
      vk::CommandBufferAllocateInfo(cmdPool, {}, 3u));

  // todo: 如果 toGpu 之后立马 backtoGpu 会有异步同步问题;
  // 但是通常来讲, 不会有人这么干... 否则无法实现 toGpu -> compute 的
  // 异步计算
  // 可能的解决方案: 把下面三步全部封装...
  memMan.toGpu(cmds[0], core, gM1.m.len * sizeof(float), gM1);
  ComputeModule::gMat<float>::axA(cmds[1], core, 42.f, gM1);
  memMan.backToCpu(cmds[2], core, gM1.m.len * sizeof(float), gM1);

  gM1.m.print();

  core.device.waitIdle();

  return 0;
}
