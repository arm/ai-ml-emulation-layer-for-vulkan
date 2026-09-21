# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

if(ANDROID)
    set(SHADER_SOURCE_DIR "tests/shader")
else()
    set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader")
endif()

# Test execution and reporting:
# Desktop tests are split by source file: mlel_cpu_tests needs no Vulkan device;
# mlel_vulkan_tests contains Vulkan tests. Add files to the appropriate target below
# so its labels and resource limits apply. Android combines both sets into
# the mlel_unit_tests executable for execution by the device test harness.
#
# On desktop, the *_GROUPED_SUITES lists below select suites to run in one CTest
# process per suite. Shared-device fixtures create one Vulkan device per suite;
# tests requiring isolation create their own devices. GoogleTest still reports
# individual cases on stdout.
# Other cases are discovered individually. Configure with
# -DVMEL_TESTS_GROUP_SUITES=OFF to run every case in a separate process instead.
#
# scripts/build.py --test -j N allows up to N concurrent CTest processes and passes
# ctest-resources.json to cap Vulkan processes at VMEL_TEST_VULKAN_WORKERS (four
# by default). CPU-labelled tests use no Vulkan slots. These labels and limits
# apply with grouping both enabled and disabled. Direct CTest runs must pass the
# resource file to enforce the Vulkan limit. For example, from <build>/tests:
#   ctest -j 56 --resource-spec-file ctest-resources.json
#   ctest -j 56 -L cpu
# Show individual case results within a grouped suite with:
#   ctest -V -R MLEmulationLayerGraphForVulkan --resource-spec-file ctest-resources.json
option(VMEL_TESTS_GROUP_SUITES "Run compatible suites in one process to share Vulkan devices" ON)
set(VMEL_TEST_VULKAN_WORKERS 4 CACHE STRING "Maximum concurrent Vulkan test processes")
if(NOT VMEL_TEST_VULKAN_WORKERS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "VMEL_TEST_VULKAN_WORKERS must be a positive integer")
endif()
# This is a concurrency budget, not a selection of physical GPU devices.
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ctest-resources.json" CONTENT
    "{\"version\":{\"major\":1,\"minor\":0},\"local\":[{\"vulkan_processes\":[{\"id\":\"shared\",\"slots\":${VMEL_TEST_VULKAN_WORKERS}}]}]}\n")

function(mlel_add_test)
    cmake_parse_arguments(ARGS "" "TARGET" "DEFINITIONS;ENVIRONMENT_MODIFICATION;GROUPED_SUITES;INCLUDE_DIRECTORIES;LIBRARIES;PROPERTIES;SOURCES" ${ARGN})

    add_executable(${ARGS_TARGET})

    target_sources(${ARGS_TARGET} PRIVATE ${ARGS_SOURCES})

    target_compile_definitions(${ARGS_TARGET} PRIVATE ${ARGS_DEFINITIONS})

    target_include_directories(${ARGS_TARGET} PRIVATE ${ARGS_INCLUDE_DIRECTORIES})

    target_link_libraries(${ARGS_TARGET} PRIVATE
        GTest::gtest_main
        ${ARGS_LIBRARIES})

    if(NOT ANDROID)
        set(DISCOVERY_PROPERTIES ${ARGS_PROPERTIES})
        foreach(MODIFICATION IN LISTS ARGS_ENVIRONMENT_MODIFICATION)
            list(APPEND DISCOVERY_PROPERTIES ENVIRONMENT_MODIFICATION "${MODIFICATION}")
        endforeach()
        set(GROUP_FILTERS)
        if(VMEL_TESTS_GROUP_SUITES AND ARGS_GROUPED_SUITES)
            foreach(SUITE IN LISTS ARGS_GROUPED_SUITES)
                set(SUITE_FILTER "${SUITE}.*")
                list(APPEND GROUP_FILTERS "${SUITE_FILTER}")
                add_test(NAME "${SUITE}"
                    COMMAND ${ARGS_TARGET} "--gtest_filter=${SUITE_FILTER}")
                if(ARGS_PROPERTIES)
                    set_tests_properties("${SUITE}" PROPERTIES ${ARGS_PROPERTIES})
                endif()
                set_property(TEST "${SUITE}" PROPERTY ENVIRONMENT_MODIFICATION "${ARGS_ENVIRONMENT_MODIFICATION}")
            endforeach()
        endif()
        # Grouped cases are registered above; discover the rest individually.
        list(JOIN GROUP_FILTERS ":" GROUP_FILTER)
        gtest_discover_tests(${ARGS_TARGET}
            TEST_FILTER "*-${GROUP_FILTER}"
            PROPERTIES ${DISCOVERY_PROPERTIES})
    endif()

endfunction()

set(MLEL_CPU_TEST_SOURCES
    common/common_tests.cpp
    graph/datatype_reference_tests.cpp
    graph/graph_ext_inst_context_tests.cpp
    graph/interval_memory_planner_tests.cpp
    graph/pipeline_cache_tests.cpp
    tensor/descriptor_binding_tests.cpp
    tensor/tensor_arm_tests.cpp
    tensor/shape_tests.cpp)
set(MLEL_VULKAN_TEST_SOURCES
    test_utils.cpp
    vulkan_test_utils.cpp
    vulkan/initialization_tests.cpp
    graph/block_match_tests.cpp
    graph/convolution_tests.cpp
    graph/datatype_tests.cpp
    graph/operator_tests.cpp
    graph/pipeline_tests.cpp
    graph/profiling_tests.cpp
    graph/properties_tests.cpp
    graph/storage_test_utils.cpp
    graph/storage_tests.cpp
    tensor/vulkan_tests.cpp)
if(NOT APPLE)
    list(APPEND MLEL_VULKAN_TEST_SOURCES graph/optical_flow_tests.cpp)
endif()

if(APPLE)
    set(MLEL_VULKAN_UNIT_TEST_ENVIRONMENT
        VK_LAYER_PATH=path_list_append:$ENV{VK_LAYER_PATH}
        DYLD_LIBRARY_PATH=path_list_append:$ENV{DYLD_LIBRARY_PATH})
else()
    set(MLEL_VULKAN_UNIT_TEST_ENVIRONMENT
        VK_LAYER_PATH=path_list_append:$<TARGET_FILE_DIR:VkLayer_Graph>
        LD_LIBRARY_PATH=path_list_append:$<TARGET_FILE_DIR:VkLayer_Graph>
        VK_LAYER_PATH=path_list_append:$<TARGET_FILE_DIR:VkLayer_Tensor>
        LD_LIBRARY_PATH=path_list_append:$<TARGET_FILE_DIR:VkLayer_Tensor>
        VK_LAYER_PATH=path_list_append:/usr/share/vulkan/explicit_layer.d)
endif()

# Group suites that share a Vulkan device. Initialization/profiling tests and
# Conv2DLargeSpatialDimension create fresh devices and remain individually
# scheduled; grouping them would serialize cases without sharing device setup.
set(MLEL_VULKAN_GROUPED_SUITES
    MLEmulationLayerGraphForVulkan
    ConvolutionRegression/Conv2DStorageFormats
    TosaConversions/TosaIntegerStorage
    TosaConversions/TosaPoolingStorage
    TosaConversions/TosaFp8Storage
    TosaConversions/TosaFp8Pad
    DatatypeCoverage/GeneratedDatatypeCast
    DatatypeCoverage/GeneratedDatatypeRescale
    DatatypeCoverage/GeneratedDatatypeLayout
    DatatypeCoverage/GeneratedDatatypeInteger)
if(NOT APPLE)
    list(APPEND MLEL_VULKAN_GROUPED_SUITES MLEmulationLayerOpticalFlowSharedDevice)
endif()

# CPU suites do not require process isolation. List single-case suites too so
# future cases automatically share the same process.
set(MLEL_CPU_GROUPED_SUITES
    MLEmulationLayerLog
    MLEmulationLayerFloat
    MLEmulationLayerUtils
    DatatypeReference
    GraphExtInstContext
    IntervalMemoryPlanner
    TensorDescriptors
    TensorARM
    TensorShape
    PipelineCache
    PipelineCache/PrecompiledShader)

if(ANDROID)
    list(APPEND MLEL_VULKAN_TEST_SOURCES ${MLEL_CPU_TEST_SOURCES})
else()
    mlel_add_test(
        TARGET mlel_cpu_tests
        GROUPED_SUITES ${MLEL_CPU_GROUPED_SUITES}
        SOURCES ${MLEL_CPU_TEST_SOURCES}
        LIBRARIES mlelutilities VkLayer_Common MLELGraph MLELTensor
        PROPERTIES LABELS cpu
    )
endif()

mlel_add_test(
    TARGET mlel_vulkan_tests
    GROUPED_SUITES ${MLEL_VULKAN_GROUPED_SUITES}
    SOURCES
        ${MLEL_VULKAN_TEST_SOURCES}
    DEFINITIONS
        SHADER_SOURCE_DIR=${SHADER_SOURCE_DIR}
    INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}
    LIBRARIES
        mlelutilities
        nlohmann_json::nlohmann_json
        VkLayer_Common
        MLELGraph
        MLELTensor
    ENVIRONMENT_MODIFICATION
        ${MLEL_VULKAN_UNIT_TEST_ENVIRONMENT}
    PROPERTIES
        LABELS vulkan RESOURCE_GROUPS "vulkan_processes:1"
)

if(ANDROID)
    # Preserve the executable name expected by the device test harness.
    set_target_properties(mlel_vulkan_tests PROPERTIES OUTPUT_NAME mlel_unit_tests)
endif()
