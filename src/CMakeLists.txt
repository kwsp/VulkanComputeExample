set(EXE_NAME main)

add_executable(${EXE_NAME}
  main.cpp
)

set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

# Vulkan
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")
find_package(Vulkan REQUIRED)

find_package(fmt CONFIG REQUIRED)

target_link_libraries(${EXE_NAME} PRIVATE
  fmt::fmt
  Vulkan::Vulkan
)
