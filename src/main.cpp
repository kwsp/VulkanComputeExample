#include <fmt/core.h>
#include <iostream>
#include <map>
#include <string>
#include <vulkan/vulkan.hpp>

struct VulkanComputeApp {
  vk::Instance instance;
  vk::PhysicalDevice physicalDevice;
  std::string physicalDeviceName;

  VulkanComputeApp() {
    // Step 1: create Vulkan instance
    createInstance();
    pickPhysicalDevice();
  }

  VulkanComputeApp(vk::Instance instance, vk::PhysicalDevice physicalDevice,
                   std::string physicalDeviceName)
      : instance(instance), physicalDevice(physicalDevice),
        physicalDeviceName(std::move(physicalDeviceName)) {}

  // Delete copy/move constructor/assignment
  VulkanComputeApp(const VulkanComputeApp &) = delete;
  VulkanComputeApp(VulkanComputeApp &&) = delete;
  VulkanComputeApp &operator=(const VulkanComputeApp &) = delete;
  const VulkanComputeApp &operator=(VulkanComputeApp &&) = delete;
  ~VulkanComputeApp() { instance.destroy(); }

  // Step 1: create Vulkan instance
  void createInstance() {
    vk::ApplicationInfo AppInfo{"VulkanCompute", VK_MAKE_VERSION(1, 0, 0),
                                "No engine", 0, VK_API_VERSION_1_1};

    // // Query vulkan layers available
    // const auto layers = vk::enumerateInstanceLayerProperties();

    // Query Vulkan extensions available
    const auto extensions = vk::enumerateInstanceExtensionProperties();
    std::cout << "available extensions:\n";
    for (const auto &extension : extensions) {
      std::cout << "\t" << extension.extensionName << "\n";
    }

    // Add VK_KHR_portability_enumeration
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_portability_enumeration.html
    // Instantiate Vulkan instance for macOS
    // https://stackoverflow.com/a/72791361/12734467
    std::vector<const char *> extensionsUsed;
    extensionsUsed.push_back("VK_KHR_portability_enumeration");

    std::cout << "Extensions used:\n";
    for (const auto &extension : extensionsUsed) {
      std::cout << "\t" << extension << "\n";
    }

    const std::vector<const char *> layers = {"VK_LAYER_KHRONOS_validation"};
    vk::InstanceCreateInfo InstanceCreateInfo(
        vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR, &AppInfo, layers,
        extensionsUsed);
    std::cout << "\n";

    instance = vk::createInstance(InstanceCreateInfo);
  }

  // Step 2: pick Vulkan physical device
  void pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> devices =
        instance.enumeratePhysicalDevices();

    if (devices.empty()) {
      throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }

    physicalDevice = devices.front();

    {
      const auto props = physicalDevice.getProperties();
      physicalDeviceName = {props.deviceName.data(), props.deviceName.size()};
    }

    fmt::println("Picked physical device '{}'.", physicalDeviceName);
  }
};

int main(int argc, char *argv[]) {
  fmt::print("Hello, world!\n");

  VulkanComputeApp app;
  return 0;
}
