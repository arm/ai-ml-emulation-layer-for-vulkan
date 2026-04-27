Tensors
=======

``VK_ARM_tensors`` is the foundation for the Emulation Layer. It provides the
tensor objects, tensor views, memory requirements, and tensor-aware shader path
used both by direct tensor compute shaders and by the graph layer.

Relationship To The Graph Layer
-------------------------------

- The tensor layer exposes ``VK_ARM_tensors`` through
  ``VK_LAYER_ML_Tensor_Emulation``.
- The graph layer uses tensor objects and tensor views from this extension.
- When both layers are enabled, the graph layer must be enabled before the
  tensor layer:

.. code-block:: shell

   export VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation

Supported Tensor API Surface
----------------------------

The tensor layer currently exposes or intercepts the following tensor-related
entry points.

Core tensor API
^^^^^^^^^^^^^^^

- ``vkCreateTensorARM``
- ``vkDestroyTensorARM``
- ``vkCreateTensorViewARM``
- ``vkDestroyTensorViewARM``
- ``vkGetTensorMemoryRequirementsARM``
- ``vkGetDeviceTensorMemoryRequirementsARM``
- ``vkBindTensorMemoryARM``
- ``vkCmdCopyTensorARM``
- ``vkGetTensorOpaqueCaptureDescriptorDataARM``
- ``vkGetTensorViewOpaqueCaptureDescriptorDataARM``

Physical-device query paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``vkGetPhysicalDeviceProperties2``
- ``vkGetPhysicalDeviceProperties2KHR``
- ``vkGetPhysicalDeviceFormatProperties2``
- ``vkGetPhysicalDeviceFeatures2``
- ``vkGetPhysicalDeviceFeatures2KHR``
- ``vkGetPhysicalDeviceExternalTensorPropertiesARM``
- ``vkGetPhysicalDeviceToolPropertiesEXT``
- ``vkCreateDevice``

Supporting Vulkan® hooks used by the tensor layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``vkCreateShaderModule``
- ``vkCreateComputePipelines``
- ``vkCreateDescriptorPool``
- ``vkCreateDescriptorSetLayout``
- ``vkUpdateDescriptorSets``
- ``vkCmdPushDescriptorSetKHR``
- ``vkCmdPipelineBarrier``
- ``vkCmdPipelineBarrier2``
- ``vkCreateImage``
- ``vkBindImageMemory``
- ``vkBindImageMemory2``
- ``vkDestroyImage``
- ``vkAllocateMemory``
- ``vkFreeMemory``
- ``vkSetDebugUtilsObjectNameEXT``

Feature And Property Model
--------------------------

The tensor layer currently reports the following behavior through standard query
paths:

- ``VkPhysicalDeviceTensorFeaturesARM.tensorNonPacked`` is enabled.
- ``VkPhysicalDeviceTensorFeaturesARM.shaderTensorAccess`` is enabled.
- ``VkPhysicalDeviceTensorFeaturesARM.tensors`` is enabled.
- ``VkPhysicalDeviceTensorFeaturesARM.shaderStorageTensorArrayDynamicIndexing``
  follows the underlying storage-buffer dynamic-indexing capability.
- ``VkPhysicalDeviceTensorFeaturesARM.shaderStorageTensorArrayNonUniformIndexing``
  follows the underlying storage-buffer non-uniform-indexing capability.
- ``VkPhysicalDeviceTensorFeaturesARM.descriptorBindingStorageTensorUpdateAfterBind``
  follows the underlying storage-buffer update-after-bind capability.

The tensor property query currently reports:

- Maximum tensor rank of ``6``.
- Tensor element, stride, and total-size limits derived from the underlying
  ``maxStorageBufferRange`` limit.
- Descriptor count limits derived from the underlying uniform-buffer descriptor
  limits.
- Tensor shader support for the compute stage only.

For tensor format queries, the layer reports transfer source, transfer
destination, tensor shader, and tensor data-graph usage on both linear and
optimal tiling.

Shader And SPIR-V™ Support
--------------------------

The tensor layer consumes ``SPV_ARM_tensors`` and rewrites tensor-aware compute
shaders to a buffer-backed implementation.

The current tensor shader path is compute-only. If a SPIR-V™ module uses
``OpTypeTensorARM`` outside a compute shader, the tensor layer rejects it.

The shared tensor and graph format handling also includes reduced-precision
formats used by the ML extensions, including:

- ``VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM``
- ``VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM``
- ``VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM``
