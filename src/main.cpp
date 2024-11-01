#include "vcm/VulkanComputeManager.hpp"
#include <fmt/core.h>
#include <vulkan/vulkan.hpp>

int main(int argc, char *argv[]) {
  fmt::print("Hello, world!\n");

  vcm::VulkanComputeManager manager;

  {
    const uint32_t N = 10;
    const uint32_t bufferSize = N * sizeof(int32_t);

    // Creating the buffers
    vk::BufferCreateInfo bufCreateInfo{vk::BufferCreateFlags(), bufferSize,
                                       vk::BufferUsageFlagBits::eStorageBuffer,
                                       vk::SharingMode::eExclusive};

    auto inBuffer = manager.get_device().createBuffer(bufCreateInfo);
    auto outBuffer = manager.get_device().createBuffer(bufCreateInfo);

    // Allocating memory
    auto inBufferMemRequirements =
        manager.get_device().getBufferMemoryRequirements(inBuffer);
    auto outBufferMemRequirements =
        manager.get_device().getBufferMemoryRequirements(outBuffer);
  }

  return 0;
}
