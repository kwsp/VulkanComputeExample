#include "vcm/VulkanComputeManager.hpp"
#include "vulkan/vulkan_core.h"
#include <fmt/core.h>
#include <fstream>
#include <stdexcept>
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

    // Load shader
    std::vector<char> shaderContents;
    const char *shaderFileName = "shaders/square.spv";
    if (std::ifstream shaderFile{shaderFileName,
                                 std::ios::binary | std::ios::ate}) {
      const size_t fileSize = shaderFile.tellg();
      shaderFile.seekg(0);
      shaderContents.resize(fileSize, '\0');
      shaderFile.read(shaderContents.data(), fileSize);
    } else {
      throw std::runtime_error(
          fmt::format("Shader object file {} not found.", shaderFileName));
    }

    vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
        vk::ShaderModuleCreateFlags(),                              // Flags
        shaderContents.size(),                                      // Code size
        reinterpret_cast<const uint32_t *>(shaderContents.data())); // Code
    vk::ShaderModule ShaderModule =
        manager.get_device().createShaderModule(shaderModuleCreateInfo);

    inBuffer.destroy(manager.get_allocator());
    outBuffer.destroy(manager.get_allocator());
  }

  return 0;
}
