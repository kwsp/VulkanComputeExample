#include "vcm/VulkanComputeManager.hpp"
#include "vulkan/vulkan_core.h"
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
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    vcm::VcmBuffer inBuffer(manager.get_allocator(), bufCreateInfo, allocInfo);
    vcm::VcmBuffer outBuffer(manager.get_allocator(), bufCreateInfo, allocInfo);

    inBuffer.destroy(manager.get_allocator());
    outBuffer.destroy(manager.get_allocator());
  }

  return 0;
}
