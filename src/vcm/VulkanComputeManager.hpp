#pragma once

#include "VmaUsage.hpp"
#include <fmt/format.h>
#include <optional>
#include <string>
#include <vulkan/vulkan.hpp>

namespace vcm {

class VulkanComputeManager {
public:
  VulkanComputeManager();

  VulkanComputeManager(const VulkanComputeManager &) = delete;
  VulkanComputeManager(VulkanComputeManager &&) = delete;
  VulkanComputeManager &operator=(const VulkanComputeManager &) = delete;
  VulkanComputeManager &operator=(VulkanComputeManager &&) = delete;

  ~VulkanComputeManager();

  static void printInstanceExtensionSupport();

  [[nodiscard]] auto &get_instance() const { return instance; }
  [[nodiscard]] auto &get_physicalDevice() const { return physicalDevice; }
  [[nodiscard]] auto &get_device() const { return device; }
  [[nodiscard]] auto &get_queue() const { return queue; }

  [[nodiscard]] auto &get_allocator() const { return m_allocator; }

  [[nodiscard]] auto &get_descriptorPool() const { return descriptorPool; }
  [[nodiscard]] auto &get_commandPool() const { return commandPool; }

  // If commandBuffer is provided, use it and only record
  // else allocate a temporary command buffer
  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                  vk::DeviceSize size,
                  vk::CommandBuffer commandBuffer = nullptr) const;

private:
  // QVulkanInstance vulkanInstance;
  vk::Instance instance;

  // Physical device
  vk::PhysicalDevice physicalDevice;
  std::string physicalDeviceName;

  // Logical device
  vk::Device device;

  // Compute queue
  vk::Queue queue;

  // Vulkan memory allocator
  VmaAllocator m_allocator;

  // Command pool
  // Manage the memory that is used to store the buffers and command buffers are
  // allocated from them
  // Command pool should be thread local
  vk::CommandPool commandPool;

  // Command buffer
  vk::CommandBuffer commandBuffer;

  // Descriptor pool
  // Discriptor sets for buffers are allocated from this
  vk::DescriptorPool descriptorPool;

  static constexpr std::array<const char *, 1> validationLayers = {
      {"VK_LAYER_KHRONOS_validation"}};

#ifdef NDEBUG
  static constexpr bool enableValidationLayers = false;
#else
  static constexpr bool enableValidationLayers = true;
#endif

  /* Create Instance */
  void createInstance();
  static bool checkValidationLayerSupport();

  /* Find physical device */
  // Select a graphics card in the system that supports the features we need.
  // Stick to the first suitable card we find.
  void pickPhysicalDevice();
  static int rateDeviceSuitability(vk::PhysicalDevice device);
  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> computeFamily;
  };
  static QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);

  /* Create logical device */
  void createLogicalDevice();

  /* Create VmaAllocator */
  void createVmaAllocator();

  /* Create command pool */
  void createCommandPool();

  /* Create command buffer */
  void createCommandBuffer();

  [[nodiscard]] vk::CommandBuffer beginTempOneTimeCommandBuffer() const {
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    auto commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

    // Immediately start recording the command buffer
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    commandBuffer.begin(beginInfo);
    return commandBuffer;
  }

  void endOneTimeCommandBuffer(vk::CommandBuffer commandBuffer) const {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    queue.submit(submitInfo);
    queue.waitIdle();
  }

  /* Create descriptor pool */
  void createDescriptorPool();

  /* Create buffers */
  [[nodiscard]] uint32_t
  findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;

  /* Cleanup */
  void cleanup();
};

// Insert a memory barrier to wait for transfer to complete before starting
// compute shader
inline void memoryBarrierTransferThenCompute(vk::CommandBuffer &commandBuffer) {
  vk::MemoryBarrier memoryBarrier{};
  memoryBarrier.srcAccessMask =
      vk::AccessFlagBits::eTransferWrite; // After copying
  memoryBarrier.dstAccessMask =
      vk::AccessFlagBits::eShaderRead; // Before compute shader reads

  commandBuffer.pipelineBarrier(
      vk::PipelineStageFlagBits::eTransfer,      // src: after the transfer op
      vk::PipelineStageFlagBits::eComputeShader, // dst: before the compute
                                                 // shader
      {}, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

// Insert a memory barrier to wait for compute shader to complete before
// transfering memory
inline void memoryBarrierComputeThenTransfer(vk::CommandBuffer &commandBuffer) {
  vk::MemoryBarrier memoryBarrier{};
  memoryBarrier.srcAccessMask =
      vk::AccessFlagBits::eShaderWrite; // After compute shader writes
  memoryBarrier.dstAccessMask =
      vk::AccessFlagBits::eTransferRead; // Before transfer reads

  commandBuffer.pipelineBarrier(
      vk::PipelineStageFlagBits::eComputeShader, // src: after compute
      vk::PipelineStageFlagBits::eTransfer,      // dst: before next transfer
      {}, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

} // namespace vcm