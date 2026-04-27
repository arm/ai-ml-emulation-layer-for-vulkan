Data Graph
==========

``VK_ARM_data_graph`` is the main graph execution path in the Emulation Layer.
TOSA, optical flow, and Motion Engine workloads are all routed through this
same graph layer rather than through separate top-level execution APIs.

Relationship To Other ML Extensions
-----------------------------------

- The graph layer requires ``VK_ARM_tensors``.
- The graph layer also requires ``VK_KHR_synchronization2``.
- Tensor objects and tensor views used by graph pipelines come from the tensor
  layer.
- ``VK_ARM_data_graph_instruction_set_tosa`` extends the core graph path for
  ``TOSA.001000.1`` workloads.
- ``VK_ARM_data_graph_optical_flow`` extends the core graph path for
  optical-flow queries and optical-flow pipeline nodes.
- ``Arm.MotionEngine.100`` is handled by the graph SPIR-V™ pass and dispatched
  through the same graph pipeline creation and execution flow.

Supported Data Graph API Surface
--------------------------------

The graph layer currently exposes or intercepts the following graph-related
entry points.

Discovery and capability query
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``vkGetPhysicalDeviceQueueFamilyProperties``
- ``vkGetPhysicalDeviceQueueFamilyProperties2``
- ``vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM``
- ``vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM``
- ``vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM``
- ``vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM``
- ``vkGetPhysicalDeviceFeatures2``
- ``vkGetPhysicalDeviceFeatures2KHR``
- ``vkGetPhysicalDeviceToolPropertiesEXT``
- ``vkCreateDevice``

Core graph pipeline and session API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``vkCreateDataGraphPipelinesARM``
- ``vkDestroyPipeline``
- ``vkGetDataGraphPipelineAvailablePropertiesARM``
- ``vkGetDataGraphPipelinePropertiesARM``
- ``vkCreateDataGraphPipelineSessionARM``
- ``vkDestroyDataGraphPipelineSessionARM``
- ``vkGetDataGraphPipelineSessionBindPointRequirementsARM``
- ``vkGetDataGraphPipelineSessionMemoryRequirementsARM``
- ``vkBindDataGraphPipelineSessionMemoryARM``
- ``vkCmdDispatchDataGraphARM``

Supporting Vulkan® hooks used by the graph layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``vkAllocateDescriptorSets``
- ``vkFreeDescriptorSets``
- ``vkUpdateDescriptorSets``
- ``vkCmdBindPipeline``
- ``vkCmdBindDescriptorSets``
- ``vkCreateTensorViewARM``
- ``vkDestroyTensorViewARM``
- ``vkCreateShaderModule``
- ``vkDestroyShaderModule``
- ``vkCmdPipelineBarrier2``
- ``vkSetDebugUtilsObjectNameEXT``

Behavior Exposed By The Graph Layer
-----------------------------------

The graph layer currently reports the following behavior through standard query
paths:

- ``VkPhysicalDeviceDataGraphFeaturesARM.dataGraph`` is enabled.
- ``VkPhysicalDeviceDataGraphFeaturesARM.dataGraphShaderModule`` is enabled.
- ``VkPhysicalDeviceDataGraphFeaturesARM.dataGraphUpdateAfterBind`` is enabled
  when the underlying device supports uniform-buffer update-after-bind.
- ``VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM.dataGraphOpticalFlow`` is
  enabled.
- Any queue family that already supports compute is also reported with
  ``VK_QUEUE_DATA_GRAPH_BIT_ARM``.

The queue-family data-graph property query currently reports two operation
types:

- ``TOSA.001000.1`` as a SPIR-V™ extended instruction set operation.
- ``OpticalFlow`` as a dedicated optical-flow operation.

For TOSA query paths, the layer reports one profile named ``Emulation Layer``
with conformant quality at TOSA level ``8K``.

TOSA Through Data Graph
-----------------------

``VK_ARM_data_graph_instruction_set_tosa`` is not a separate execution API. It
extends ``VK_ARM_data_graph`` so that graph pipelines can consume the
``TOSA.001000.1`` SPIR-V™ extended instruction set.

The graph pass currently implements the following ``TOSA.001000.1`` operations:

- Elementwise and logical: ``ABS``, ``ADD``, ``BITWISE_AND``,
  ``BITWISE_NOT``, ``BITWISE_OR``, ``BITWISE_XOR``, ``EQUAL``, ``GREATER``,
  ``GREATER_EQUAL``, ``INTDIV``, ``LOGICAL_AND``, ``LOGICAL_LEFT_SHIFT``,
  ``LOGICAL_NOT``, ``LOGICAL_OR``, ``LOGICAL_RIGHT_SHIFT``,
  ``LOGICAL_XOR``, ``MAXIMUM``, ``MINIMUM``, ``MUL``, ``NEGATE``, ``POW``,
  ``SUB``.
- Unary math and activations: ``CAST``, ``CEIL``, ``CLAMP``, ``CLZ``, ``COS``,
  ``ERF``, ``EXP``, ``FLOOR``, ``LOG``, ``RECIPROCAL``, ``RSQRT``,
  ``SIGMOID``, ``SIN``, ``TANH``.
- Reductions and selection: ``ARGMAX``, ``REDUCE_ALL``, ``REDUCE_ANY``,
  ``REDUCE_MAX``, ``REDUCE_MIN``, ``REDUCE_PRODUCT``, ``REDUCE_SUM``,
  ``SELECT``.
- Convolution, pooling, and matrix-style operations: ``AVG_POOL2D``,
  ``CONV2D``, ``CONV3D``, ``DEPTHWISE_CONV2D``, ``MATMUL``, ``MAX_POOL2D``,
  ``TRANSPOSE_CONV2D``.
- Data movement and layout: ``ARITHMETIC_RIGHT_SHIFT``, ``CONCAT``, ``FFT2D``,
  ``GATHER``, ``PAD``, ``RESCALE``, ``RESHAPE``, ``RESIZE``, ``REVERSE``,
  ``RFFT2D``, ``SCATTER``, ``SLICE``, ``TABLE``, ``TILE``, ``TRANSPOSE``.

Motion Engine Through Data Graph
--------------------------------

``Arm.MotionEngine.100`` is also handled by the graph SPIR-V™ path rather than
by a separate Vulkan® execution API.

The graph pass currently implements the following Motion Engine operations:

- ``MIN_SAD``
- ``MIN_SAD_COST``
- ``RAW_SAD``

Optical Flow Through Data Graph
-------------------------------

``VK_ARM_data_graph_optical_flow`` extends ``VK_ARM_data_graph`` rather than
introducing a separate execution API. Optical-flow pipelines are still
created, queried, and dispatched through the core data-graph pipeline and
session API listed above.

Capability query
^^^^^^^^^^^^^^^^

The optical-flow path currently reports and accepts:

- Output and hint grid sizes ``1x1``, ``2x2``, ``4x4``, and ``8x8``.
- Hint input support.
- Cost output support.
- Image sizes from ``64x64`` up to ``8192x8192``.
- Input image formats:
  ``VK_FORMAT_R8G8B8_UNORM``, ``VK_FORMAT_B8G8R8_UNORM``,
  ``VK_FORMAT_R8G8B8A8_UNORM``, ``VK_FORMAT_B8G8R8A8_UNORM``,
  ``VK_FORMAT_B10G11R11_UFLOAT_PACK32``, ``VK_FORMAT_R8_UNORM``.
- Flow-vector output format ``VK_FORMAT_R16G16_SFLOAT``.
- Cost output format ``VK_FORMAT_R16_UINT``.

Creation constraints
^^^^^^^^^^^^^^^^^^^^

The current optical-flow implementation accepts:

- ``VK_DATA_GRAPH_OPTICAL_FLOW_CREATE_ENABLE_HINT_BIT_ARM``
- ``VK_DATA_GRAPH_OPTICAL_FLOW_CREATE_ENABLE_COST_BIT_ARM``
- Performance levels ``SLOW``, ``MEDIUM``, ``FAST``, and ``UNKNOWN``
- Session create flag
  ``VK_DATA_GRAPH_PIPELINE_SESSION_CREATE_OPTICAL_FLOW_CACHE_BIT_ARM``

When hint input is enabled, the hint grid size must match the selected output
grid size.
