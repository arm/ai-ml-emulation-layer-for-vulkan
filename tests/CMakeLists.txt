# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

if(ANDROID)
    set(SHADER_SOURCE_DIR "/data/local/tmp/vmel/shader")
else()
    set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader")
endif()

function(mlel_add_test)
    cmake_parse_arguments(ARGS "" "TARGET" "DEFINITIONS;INCLUDE_DIRECTORIES;LIBRARIES;PROPERTIES;SOURCES" ${ARGN})

    add_executable(${ARGS_TARGET})

    target_sources(${ARGS_TARGET} PRIVATE ${ARGS_SOURCES})

    target_compile_definitions(${ARGS_TARGET} PRIVATE ${ARGS_DEFINITIONS})

    target_include_directories(${ARGS_TARGET} PRIVATE ${ARGS_INCLUDE_DIRECTORIES})

    target_link_libraries(${ARGS_TARGET} PRIVATE
        GTest::gtest_main
        ${ARGS_LIBRARIES})

    if(NOT ANDROID)
        gtest_discover_tests(${ARGS_TARGET}
            PROPERTIES ${ARGS_PROPERTIES})
    endif()

endfunction()

if(NOT APPLE)
    mlel_add_test(
        TARGET mlel_unit_tests
        SOURCES
            vulkan.cpp
        DEFINITIONS
            SHADER_SOURCE_DIR=${SHADER_SOURCE_DIR}
        LIBRARIES
            mlelutilities
            VkLayer_Common
        PROPERTIES
            ENVIRONMENT_MODIFICATION VK_LAYER_PATH=path_list_append:$<TARGET_FILE_DIR:VkLayer_Graph>
            ENVIRONMENT_MODIFICATION LD_LIBRARY_PATH=path_list_append:$<TARGET_FILE_DIR:VkLayer_Graph>

            ENVIRONMENT_MODIFICATION VK_LAYER_PATH=path_list_append:$<TARGET_FILE_DIR:VkLayer_Tensor>
            ENVIRONMENT_MODIFICATION LD_LIBRARY_PATH=path_list_append:$<TARGET_FILE_DIR:VkLayer_Tensor>

            ENVIRONMENT_MODIFICATION VK_LAYER_PATH=path_list_append:/usr/share/vulkan/explicit_layer.d
    )
else()
    mlel_add_test(
        TARGET mlel_unit_tests
        SOURCES
            vulkan.cpp
        DEFINITIONS
            SHADER_SOURCE_DIR=${SHADER_SOURCE_DIR}
        LIBRARIES
            mlelutilities
            VkLayer_Common
        PROPERTIES
            ENVIRONMENT_MODIFICATION VK_LAYER_PATH=path_list_append:$ENV{VK_LAYER_PATH}
            ENVIRONMENT_MODIFICATION DYLD_LIBRARY_PATH=path_list_append:$ENV{DYLD_LIBRARY_PATH}
    )
endif()
