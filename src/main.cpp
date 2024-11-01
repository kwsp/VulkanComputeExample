#include <fmt/core.h>
#include <iostream>
#include <vulkan/vulkan.hpp>

struct VulkanComputeApp {
  vk::Instance instance;

  VulkanComputeApp() {
    // Step 1: create Vulkan instance
    createInstance();
  }

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

    instance = vk::createInstance(InstanceCreateInfo);
  }
};

int main(int argc, char *argv[]) {
  fmt::print("Hello, world!\n");

  VulkanComputeApp app;
  return 0;
}
