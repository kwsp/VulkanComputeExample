#pragma once

#include "VmaUsage.hpp"
#include <filesystem>
#include <fmt/format.h>
#include <vulkan/vulkan.hpp>

namespace vcm {

namespace fs = std::filesystem;

/*
Helpers for vk-hpp to vk
*/

inline auto toVk(const vk::BufferCreateInfo *createInfo) {
  return reinterpret_cast<VkBufferCreateInfo const *>(createInfo);
}

} // namespace vcm