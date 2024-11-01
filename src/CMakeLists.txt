set(EXE_NAME main)

add_executable(${EXE_NAME}
  main.cpp
  vcm/VulkanComputeManager.hpp
  vcm/VulkanComputeManager.cpp
)

set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

# Vulkan
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")
find_package(Vulkan REQUIRED)

# Vulkan validation layers from vcpkg
set(ENV{VK_ADD_LAYER_PATH} "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin")

# Vulkan memory allocator
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

find_package(fmt CONFIG REQUIRED)

target_link_libraries(${EXE_NAME} PRIVATE
  fmt::fmt
  Vulkan::Vulkan
  GPUOpen::VulkanMemoryAllocator
)
