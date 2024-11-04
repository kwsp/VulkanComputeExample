#include "Shader.hpp"
#include <fmt/format.h>
#include <fstream>

namespace vcm {

auto loadShader(vk::Device device, const char *shaderFileName)
    -> vk::ShaderModule {

  // Load shader
  std::vector<char> shaderContents;
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
      vk::ShaderModuleCreateFlags(), shaderContents.size(),
      reinterpret_cast<const uint32_t *>(shaderContents.data())); // NOLINT
  return device.createShaderModule(shaderModuleCreateInfo);
}

} // namespace vcm