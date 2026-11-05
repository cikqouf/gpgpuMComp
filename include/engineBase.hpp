#pragma once

#include <iostream>
#include <string>

#include <vector>
#include <vulkan/vulkan_raii.hpp>

namespace EngineBase {

static uint32_t memTypeChoose(const vk::raii::PhysicalDevice &phyDevice,
                              vk::MemoryPropertyFlags cond) {
  auto memTypes = phyDevice.getMemoryProperties();
  for (int idx = 0; idx < memTypes.memoryTypeCount; ++idx) {
    if ((cond & memTypes.memoryTypes[idx].propertyFlags) == cond)
      return idx;
  }

  std::cerr << "EngineBase::memTypeChoose: fallback to 0u...\n";
  return 0u;
}

class initBase {
private:
  void checkLayers(const char *name) const noexcept {
    auto layerP = context.enumerateInstanceLayerProperties();

    for (int i = 0; i < layerP.size(); ++i) {
      if (std::string(name) == std::string(layerP[i].layerName)) {
        std::cerr << name << " is ok\n";
        return;
      }
    }

    std::cerr << name << " is not ok, layer list below\n";

    for (int i = 0; i < layerP.size(); ++i) {
      std::cerr << layerP[i].layerName << "\n";
    }
  }

  void checkExtensions(const char *name) const noexcept {
    auto extensionP = context.enumerateInstanceExtensionProperties();

    for (int i = 0; i < extensionP.size(); ++i) {
      if (std::string(name) == std::string(extensionP[i].extensionName)) {
        std::cerr << name << " is ok\n";
        return;
      }
    }

    std::cerr << name << " is not ok, extension list below\n";

    for (int i = 0; i < extensionP.size(); ++i) {
      std::cerr << extensionP[i].extensionName << "\n";
    }
  }

  uint32_t chooseQueueFamilyIndex(vk::QueueFlags cond) const noexcept {

    auto qps = physicalDevice.getQueueFamilyProperties();
    uint32_t idx = 0u;
    for (const auto &qp : qps) {
      if ((qp.queueFlags & cond) == cond) {
        return idx;
      }
      ++idx;
    }

    std::cerr << "chooseQueueFamilyIndex fallback to 0u\n";
    return 0u;
  }

  static vk::Bool32
  debugCallBack(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
                const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData) {
    std::cerr << pCallbackData->pMessageIdName << " " << pCallbackData->pMessage
              << "\n";
    return vk::False;
  }

private:
  vk::raii::Context context = vk::raii::Context();
  vk::raii::Instance instance = nullptr;

  vk::raii::DebugUtilsMessengerEXT messenger = nullptr;

public:
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;
  uint32_t queueFamilyIndex = 0u;

public:
  initBase() = delete;

  initBase(vk::PhysicalDeviceType phyDeviceType)
      : instance([&] {
          const vk::ApplicationInfo appInfo(
              "computeApp", vk::makeVersion(0, 0, 2), "computeCore",
              vk::makeVersion(0, 0, 2), vk::makeApiVersion(0, 1, 2, 0));

          checkLayers("VK_LAYER_KHRONOS_validation");
          std::vector<const char *> layerNames = {
              "VK_LAYER_KHRONOS_validation"};

          checkExtensions("VK_EXT_debug_utils");
          std::vector<const char *> extensionNames = {"VK_EXT_debug_utils"};

          return context.createInstance(vk::InstanceCreateInfo(
              {}, &appInfo, layerNames.size(), layerNames.data(),
              extensionNames.size(), extensionNames.data()));
        }()),
        messenger(instance.createDebugUtilsMessengerEXT(
            vk::DebugUtilsMessengerCreateInfoEXT(
                {},
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                    vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
                debugCallBack))) {

    auto physicalDevices = instance.enumeratePhysicalDevices();
    for (const auto &phyDevice : physicalDevices) {
      if (phyDevice.getProperties().deviceType == phyDeviceType) {
        physicalDevice = phyDevice;
        break;
      }
    }

    float queuePrioties[] = {1.f};
    queueFamilyIndex = chooseQueueFamilyIndex(vk::QueueFlagBits::eCompute |
                                              vk::QueueFlagBits::eTransfer);

    const vk::DeviceQueueCreateInfo queueInfo({}, queueFamilyIndex, 1,
                                              queuePrioties);
    device = physicalDevice.createDevice(
        vk::DeviceCreateInfo({}, 1u, &queueInfo, 0, nullptr, 0, nullptr));
  }
};

} // namespace EngineBase
