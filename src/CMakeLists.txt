set(EXE_NAME main)

# Vulkan
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")
find_package(Vulkan REQUIRED)

# Vulkan validation layers from vcpkg
set(ENV{VK_ADD_LAYER_PATH} "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin")

# Vulkan memory allocator
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

# Other dependencies
find_package(fmt CONFIG REQUIRED)

add_executable(${EXE_NAME}
  main.cpp

  vcm/VulkanComputeManager.hpp
  vcm/VulkanComputeManager.cpp
  vcm/VmaUsage.hpp
  vcm/VmaUsage.cpp
  vcm/Buffer.hpp
  vcm/Common.hpp
  vcm/Shader.hpp
  vcm/Shader.cpp
)

set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)


if (NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR "VULKAN_SDK environment variable is not set.")
endif()


function(vcm_add_hlsl_shaders TARGET)
    # Create shader binary directory
    set(SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${SHADER_BINARY_DIR}")

    # Add a post-build command to copy "BINARY_DIR/shaders" to "BINARY_DIR/CONFIG/shaders"
    add_custom_command(
        OUTPUT "${SHADER_BINARY_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${SHADER_BINARY_DIR}"
            "$<TARGET_FILE_DIR:${TARGET}>/shaders"
        COMMENT "Copying shader objects to binary config dir"
    )
    add_custom_target(vcm_copy_shaders_target DEPENDS ${SHADER_BINARY_DIR})
    add_dependencies(${TARGET} vcm_copy_shaders_target)

    message(STATUS "ARGN: ${ARGN}")

    # Iterate over each shader file and process it
    foreach(SHADER_SOURCE_FILE ${ARGN})
        set(SHADER_SOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE_FILE}")

        # Get the base name of the shader file
        cmake_path(GET SHADER_SOURCE_FILE STEM SHADER_STEM)
        set(SHADER_OUTPUT_FILE "${SHADER_BINARY_DIR}/${SHADER_STEM}.spv")

        message(STATUS "SHADER_SOURCE_FILE: ${SHADER_SOURCE_FILE}")
        message(STATUS "SHADER_OUTPUT_FILE: ${SHADER_OUTPUT_FILE}")

        # Compile HLSL -> SPIR-V
        add_custom_command(
            OUTPUT "${SHADER_OUTPUT_FILE}"
            COMMAND $ENV{VULKAN_SDK}/bin/dxc -T cs_6_0 -E "Main" -spirv -fvk-use-dx-layout -fspv-target-env=vulkan1.1 -Fo "${SHADER_OUTPUT_FILE}" "${SHADER_SOURCE_FILE}"
            DEPENDS "${SHADER_SOURCE_FILE}"
            WORKING_DIRECTORY ${SHADER_BINARY_DIR}
            COMMENT "Building Shader ${SHADER_SOURCE_FILE}"
        )

        # Add the shader target as a dependency to the main target
        set(SHADER_TARGET_NAME "vcm_shader_${SHADER_STEM}")
        add_custom_target(${SHADER_TARGET_NAME} DEPENDS ${SHADER_OUTPUT_FILE})
        message(STATUS "added custom target ${SHADER_TARGET_NAME}")
        add_dependencies(vcm_copy_shaders_target ${SHADER_TARGET_NAME})
    endforeach()
endfunction()


vcm_add_hlsl_shaders(${EXE_NAME} 
  shaders/square.hlsl
  shaders/add.hlsl
)


target_link_libraries(${EXE_NAME} PRIVATE
  fmt::fmt
  Vulkan::Vulkan
  GPUOpen::VulkanMemoryAllocator
)

