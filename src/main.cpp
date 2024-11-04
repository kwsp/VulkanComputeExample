#include "vcm/Buffer.hpp"
#include "vcm/Shader.hpp"
#include "vcm/VulkanComputeManager.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <numeric>
#include <stdexcept>
#include <vk_mem_alloc.h>
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
    allocInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    vcm::VcmBuffer inBuffer(manager.get_allocator(), bufCreateInfo, allocInfo);
    vcm::VcmBuffer outBuffer(manager.get_allocator(), bufCreateInfo, allocInfo);

    // Copy data to inBuffer
    std::vector<uint32_t> inData(N);
    std::iota(inData.begin(), inData.end(), 0);
    fmt::println("In data:\t{}", fmt::join(inData, ", "));

    vmaCopyMemoryToAllocation(manager.get_allocator(), inData.data(),
                              inBuffer.allocation, 0, N * sizeof(uint32_t));

    // Load shader
    auto shader = vcm::loadShader(manager.get_device(), "shaders/square.spv");

    /*
    Create the compute pipeline
    */
    // 1. Descriptor set layer
    // layout of data to be passed into the pipeline
    // (this is not the actual descriptor set, just the layout)
    // Specified using a series of DescriptorSetLayoutBinding objects
    const std::vector<vk::DescriptorSetLayoutBinding>
        descriptorSetLayoutBinding = {
            {0, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
        };

    vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
        vk::DescriptorSetLayoutCreateFlags(), descriptorSetLayoutBinding);

    auto descriptorSetLayout = manager.get_device().createDescriptorSetLayout(
        descriptorSetLayoutCreateInfo);

    // 2. Pipeline layout
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
        vk::PipelineLayoutCreateFlags(), descriptorSetLayout);
    auto pipelineLayout =
        manager.get_device().createPipelineLayout(pipelineLayoutCreateInfo);
    auto pipelineCache =
        manager.get_device().createPipelineCache(vk::PipelineCacheCreateInfo());

    // 3. Create the Pipeline
    vk::PipelineShaderStageCreateInfo pipelineShaderCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
        shader, "Main");

    vk::ComputePipelineCreateInfo computePipelineCreateInfo(
        vk::PipelineCreateFlags(), pipelineShaderCreateInfo, pipelineLayout);

    // TODO save and load the pipelineCache at the start and end of app
    // https://docs.vulkan.org/samples/latest/samples/performance/hpp_pipeline_cache/README.html#_vulkan_pipeline_cache
    auto computePipeline_ = manager.get_device().createComputePipeline(
        pipelineCache, computePipelineCreateInfo);
    if (computePipeline_.result != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to create commpute pipeline.");
    }
    vk::Pipeline computePipeline = computePipeline_.value;

    // 4. Create the DescriptorSet
    vk::DescriptorSetAllocateInfo descriptorSetAllocInfo(
        manager.get_descriptorPool(), 1, &descriptorSetLayout);
    const std::vector<vk::DescriptorSet> descriptorSets =
        manager.get_device().allocateDescriptorSets(descriptorSetAllocInfo);
    auto descriptorSet = descriptorSets.front();
    vk::DescriptorBufferInfo inBufferInfo(inBuffer.buffer, 0,
                                          N * sizeof(int32_t));
    vk::DescriptorBufferInfo outBufferInfo(outBuffer.buffer, 0,
                                           N * sizeof(int32_t));

    const std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
        {descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
         &inBufferInfo},
        {descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
         &outBufferInfo},
    };
    manager.get_device().updateDescriptorSets(writeDescriptorSets, {});

    /*
    Submitting work to the GPU
    */

    // 1. Allocate command buffer
    vk::CommandBufferAllocateInfo commandBufferAllocInfo(
        manager.get_commandPool(), vk::CommandBufferLevel::ePrimary, 1);
    const auto cmdBuffers =
        manager.get_device().allocateCommandBuffers(commandBufferAllocInfo);
    auto cmdBuffer = cmdBuffers.front();

    // 2. Recording commands
    // Bind the pipeline and descriptorset, and record a dispatch call
    vk::CommandBufferBeginInfo cmdBufferBeginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmdBuffer.begin(cmdBufferBeginInfo);
    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
    cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 pipelineLayout, 0, {descriptorSet}, {});
    // Record the number of threads to launch in the device.
    // Here we launch 1 thread per element
    cmdBuffer.dispatch(N, 1, 1);
    cmdBuffer.end();

    // 3. Submit to GPU
    // First get the Queue from the Device
    // Create a Fence (a mechanism to wait for the compute shader to complete)
    vk::Fence fence = manager.get_device().createFence(vk::FenceCreateInfo());

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &cmdBuffer);
    manager.get_queue().submit({submitInfo}, fence);
    auto result =
        manager.get_device().waitForFences({fence}, true, uint64_t(-1));

    if (result != vk::Result::eSuccess) {
      throw std::runtime_error(
          fmt::format("vk result = {}", static_cast<int32_t>(result)));
    }

    // Finally, read results
    std::vector<uint32_t> outData(N);
    vmaCopyAllocationToMemory(manager.get_allocator(), outBuffer.allocation, 0,
                              outData.data(), N * sizeof(uint32_t));

    fmt::println("Out data:\t{}", fmt::join(outData, ", "));

    /*
    Cleanup
    */
    inBuffer.destroy(manager.get_allocator());
    outBuffer.destroy(manager.get_allocator());
  }

  return 0;
}
