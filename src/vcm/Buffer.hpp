#pragma once

#include "Common.hpp"
#include "VmaUsage.hpp"

namespace vcm {

/*
Vma allocation buffer
*/
struct VcmBuffer {
  VkBuffer buffer{};
  VmaAllocation allocation{};

  VcmBuffer(VmaAllocator allocator, const vk::BufferCreateInfo &createInfo,
            const VmaAllocationCreateInfo &allocInfo) {

    // Creating the buffers
    vmaCreateBuffer(allocator, vcm::toVk(&createInfo), &allocInfo, &buffer,
                    &allocation, nullptr);
  }

  void destroy(VmaAllocator allocator) {
    vmaDestroyBuffer(allocator, buffer, allocation);
  }
};

} // namespace vcm