#pragma once

#include "Common.hpp"
#include "VmaUsage.hpp"
#include <filesystem>
#include <fmt/format.h>
#include <optional>
#include <span>
#include <string>
#include <vector>
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

  // Create buffer helpers
  [[nodiscard]] VulkanBuffer
  createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags properties) const;

  [[nodiscard]] VulkanBuffer createStagingBufferSrc(vk::DeviceSize size) const {
    return createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
                        vk::MemoryPropertyFlagBits::eHostVisible |
                            vk::MemoryPropertyFlagBits::eHostCoherent);
  }
  [[nodiscard]] VulkanBuffer createStagingBufferDst(vk::DeviceSize size) const {
    return createBuffer(size, vk::BufferUsageFlagBits::eTransferDst,
                        vk::MemoryPropertyFlagBits::eHostVisible |
                            vk::MemoryPropertyFlagBits::eHostCoherent);
  }

  [[nodiscard]] VulkanBuffer createDeviceBufferSrc(vk::DeviceSize size) const {
    return createBuffer(size,
                        vk::BufferUsageFlagBits::eTransferSrc |
                            vk::BufferUsageFlagBits::eStorageBuffer,
                        vk::MemoryPropertyFlagBits::eDeviceLocal);
  }

  [[nodiscard]] VulkanBuffer createDeviceBufferDst(vk::DeviceSize size) const {
    return createBuffer(size,
                        vk::BufferUsageFlagBits::eTransferDst |
                            vk::BufferUsageFlagBits::eStorageBuffer,
                        vk::MemoryPropertyFlagBits::eDeviceLocal);
  }

  [[nodiscard]] VulkanImage createImage2D(uint32_t width, uint32_t height,
                                          vk::Format format,
                                          vk::ImageUsageFlags usage) const;

  // If commandBuffer is provided, use it and only record
  // else allocate a temporary command buffer
  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                  vk::DeviceSize size,
                  vk::CommandBuffer commandBuffer = nullptr) const;

  struct CopyBufferT {
    vk::Buffer src;
    vk::Buffer dst;
    vk::DeviceSize size;
  };
  void copyBuffers(std::span<CopyBufferT> buffersToCopy) const;

  void copyBufferToImage(vk::Buffer buffer, vk::Image image,
                         uint32_t imageWidth, uint32_t imageHeight,
                         vk::CommandBuffer commandBuffer = nullptr) const;

  void copyImageToBuffer(vk::Image image, vk::Buffer buffer, uint32_t width,
                         uint32_t height,
                         vk::CommandBuffer commandBuffer) const;

  // Transfer data from a user host buffer to a Vulkan staging buffer
  template <typename T>
  void copyToStagingBuffer(std::span<const T> data,
                           vk::DeviceMemory memory) const {
    vk::DeviceSize size = data.size() * sizeof(T);
    // Step 1: Map the memory associated with the staging buffer
    void *mappedMemory = device.mapMemory(memory, 0, size);
    // Step 2: Copy data from the vector to the mapped memory
    memcpy(mappedMemory, data.data(), static_cast<size_t>(size));
    // Step 3: Unmap the memory so the GPU can access it
    device.unmapMemory(memory);
  }
  template <typename T>
  void copyToStagingBuffer(std::span<const T> data,
                           VulkanBuffer &stagingBuffer) const {
    copyToStagingBuffer(data, stagingBuffer.memory.get());
  }
  template <typename T>
  void copyToStagingBuffer(std::span<const T> data,
                           VulkanBufferRef &stagingBuffer) const {
    copyToStagingBuffer(data, stagingBuffer.memory);
  }

  template <typename T>
  void copyFromStagingBuffer(vk::DeviceMemory memory, std::span<T> data) const {
    vk::DeviceSize size = data.size() * sizeof(T);
    void *mappedMemory = device.mapMemory(memory, 0, size);
    // Copy the data from GPU memory to a local buffer
    memcpy(data.data(), mappedMemory, static_cast<size_t>(size));
    // Unmap the memory after retrieving the data
    device.unmapMemory(memory);
  }
  template <typename T>
  void copyFromStagingBuffer(const VulkanBuffer &stagingBuffer,
                             std::span<T> data) const {
    copyFromStagingBuffer<T>(stagingBuffer.memory.get(), data);
  }
  template <typename T>
  void copyFromStagingBuffer(const VulkanBufferRef &stagingBuffer,
                             std::span<T> data) const {
    copyFromStagingBuffer<T>(stagingBuffer.memory, data);
  }

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

  /* Load shaders */
  [[nodiscard]] vk::UniqueShaderModule
  loadShader(const fs::path &filename) const;
  [[nodiscard]] vk::UniqueShaderModule
  createShaderModule(const std::vector<char> &computeShaderCode) const;

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