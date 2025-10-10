# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

###############################################################################
# Graph layer
###############################################################################

add_subdirectory(tosa)

add_library(VkLayer_Graph SHARED)

add_library(${ML_SDK_EL_NAMESPACE}::VkLayer_Graph ALIAS VkLayer_Graph)

if(WIN32)
  # Ensure the Windows DLL exports the exact Vulkan loader entry points
  target_sources(VkLayer_Graph PRIVATE VkLayer_Graph.def)
endif()

target_compile_options(VkLayer_Graph PRIVATE ${VMEL_COMPILE_OPTIONS})

target_sources(VkLayer_Graph PRIVATE
    compute.cpp
    graph_layer.cpp
    graph_log.cpp
    memory_planner.cpp
    pipeline_cache.cpp
    spirv_pass.cpp
    spirv_pass_tosaspv_v100.cpp
    tensor.cpp)

# Generated GLSL shaders
add_dependencies(VkLayer_Graph
    mlel_generate_glsl_header)

# Pre compiled SPIR-V modules
add_dependencies(VkLayer_Graph
    mlel_generate_spv_header)

target_link_libraries(VkLayer_Graph PRIVATE
    VkLayer_Common)

target_include_directories(VkLayer_Graph PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS VkLayer_Graph EXPORT ${ML_SDK_EL_PACKAGE_NAME}Config)

# Generate VkLayer_Graph.json
set(GRAPH_LIBRARY_PATH $<TARGET_FILE_NAME:VkLayer_Graph>)
if(WIN32)
    set(GRAPH_LIBRARY_PATH ".\\\\${GRAPH_LIBRARY_PATH}")
endif()

configure_file(layer.d/VkLayer_Graph.json.in layer.d/VkLayer_Graph.json.in)
file(GENERATE
    OUTPUT $<TARGET_FILE_DIR:VkLayer_Graph>/VkLayer_Graph.json
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/layer.d/VkLayer_Graph.json.in)

if(WIN32)
    install(FILES $<TARGET_FILE_DIR:VkLayer_Graph>/VkLayer_Graph.json TYPE BIN)
else()
    install(FILES $<TARGET_FILE_DIR:VkLayer_Graph>/VkLayer_Graph.json DESTINATION share/vulkan/explicit_layer.d)
endif()
