#pragma once

#include <vulkan/vulkan.hpp>

namespace vcm {

// Load a compiled spv shader object
auto loadShader(vk::Device device, const char *shaderFileName)
    -> vk::ShaderModule;

} // namespace vcm