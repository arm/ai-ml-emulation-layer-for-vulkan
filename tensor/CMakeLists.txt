# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

add_library(VkLayer_Tensor SHARED)

add_library(${ML_SDK_EL_NAMESPACE}::VkLayer_Tensor ALIAS VkLayer_Tensor)

target_set_vulkan_exports(VkLayer_Tensor)

target_compile_options(VkLayer_Tensor PRIVATE ${VMEL_COMPILE_OPTIONS})

target_sources(VkLayer_Tensor PRIVATE
    tensor_layer.cpp
    tensor_processor.cpp
    tensor_arm.cpp
    tensor_log.cpp
    tensor_view.cpp
    tensor_descriptor.cpp
    spirv_glsl_tensor_buffer.cpp)

target_link_libraries(VkLayer_Tensor PRIVATE
    VkLayer_Common
    spirv-cross-core
    spirv-cross-glsl)

install(TARGETS VkLayer_Tensor EXPORT ${ML_SDK_EL_PACKAGE_NAME}Config)

# Generate VkLayer_Tensor.json
set(TENSOR_LIBRARY_PATH $<TARGET_FILE_NAME:VkLayer_Tensor>)
if(WIN32)
    set(TENSOR_LIBRARY_PATH ".\\\\${TENSOR_LIBRARY_PATH}")
elseif(APPLE)
    set(TENSOR_LIBRARY_PATH "../../../lib/${TENSOR_LIBRARY_PATH}")
endif()

configure_file(layer.d/VkLayer_Tensor.json.in layer.d/VkLayer_Tensor.json.in)
file(GENERATE
    OUTPUT $<TARGET_FILE_DIR:VkLayer_Tensor>/VkLayer_Tensor.json
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/layer.d/VkLayer_Tensor.json.in)

if(WIN32)
    install(FILES $<TARGET_FILE_DIR:VkLayer_Tensor>/VkLayer_Tensor.json TYPE BIN)
else()
    install(FILES $<TARGET_FILE_DIR:VkLayer_Tensor>/VkLayer_Tensor.json DESTINATION share/vulkan/explicit_layer.d)
endif()
