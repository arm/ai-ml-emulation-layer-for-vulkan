#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

set(VULKAN_LAYER_EXPORTS
    vkNegotiateLoaderLayerInterfaceVersion
    vkGetInstanceProcAddr
    vkGetDeviceProcAddr
    vkEnumerateInstanceLayerProperties
    vk_layerGetPhysicalDeviceProcAddr
)

if(ANDROID)
    list(APPEND VULKAN_LAYER_EXPORTS
        vkEnumerateInstanceExtensionProperties
        vkEnumerateDeviceLayerProperties
        vkEnumerateDeviceExtensionProperties
    )
endif()

function(target_set_vulkan_exports TARGET_NAME)
    if(WIN32)
        set(DEF_CONTENT "LIBRARY ${TARGET_NAME}\nEXPORTS\n")
        foreach(SYMBOL ${VULKAN_LAYER_EXPORTS})
            string(APPEND DEF_CONTENT "   ${SYMBOL}\n")
        endforeach()

        set(DEF_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.def")
        file(WRITE "${DEF_FILE}" "${DEF_CONTENT}")

        target_sources(${TARGET_NAME} PRIVATE ${DEF_FILE})

    elseif(APPLE)
        set(EXP_CONTENT "")
        foreach(SYMBOL ${VULKAN_LAYER_EXPORTS})
            string(APPEND EXP_CONTENT "_${SYMBOL}\n")
        endforeach()

        set(EXP_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.exp")
        file(WRITE "${EXP_FILE}" "${EXP_CONTENT}")

        target_link_options(${TARGET_NAME} PRIVATE
            "LINKER:-exported_symbols_list,${EXP_FILE}"
        )

    elseif(UNIX)
        set(MAP_CONTENT "{ \n    global:\n")
        foreach(SYMBOL ${VULKAN_LAYER_EXPORTS})
            string(APPEND MAP_CONTENT "        ${SYMBOL};\n")
        endforeach()
        string(APPEND MAP_CONTENT "    local: *;\n};\n")

        set(MAP_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.map")
        file(WRITE "${MAP_FILE}" "${MAP_CONTENT}")

        target_link_options(${TARGET_NAME} PRIVATE
            "LINKER:--version-script=${MAP_FILE}"
            "LINKER:--exclude-libs,ALL"
        )
    endif()

endfunction()
