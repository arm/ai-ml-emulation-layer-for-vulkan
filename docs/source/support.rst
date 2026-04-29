ML Extension Overview
=====================

The ML Emulation Layer for Vulkan® exposes ML functionality through two Vulkan® layers:

- ``VK_LAYER_ML_Tensor_Emulation``, which provides tensor objects, tensor views,
  tensor queries, and tensor-shader emulation through ``VK_ARM_tensors``.
- ``VK_LAYER_ML_Graph_Emulation``, which provides graph execution through
  ``VK_ARM_data_graph`` and extends that same graph path for TOSA and optical
  flow.

The tensor layer is the foundation. The graph layer builds on top of it and is
the route used for generic data-graph execution, TOSA workloads, optical flow
workloads, and Motion Engine operations.

API Flow
--------

.. code-block:: text

   Direct tensor flow
   Application
     -> enable VK_LAYER_ML_Tensor_Emulation
     -> use VK_ARM_tensors
     -> create tensors and tensor views
     -> submit compute shaders using SPV_ARM_tensors

   Data graph flow
   Application
     -> enable VK_LAYER_ML_Graph_Emulation
     -> enable VK_LAYER_ML_Tensor_Emulation
     -> use VK_ARM_data_graph
     -> graph layer uses VK_ARM_tensors tensor objects and tensor views
     -> create and dispatch graph pipelines from SPV_ARM_graph
     -> optional graph paths
        -> VK_ARM_data_graph_instruction_set_tosa
           -> TOSA.001000.1
        -> VK_ARM_data_graph_optical_flow
           -> optical-flow query paths and optical-flow pipeline nodes
        -> Arm.MotionEngine.100
           -> Motion Engine operations through the same data-graph path

In general, the execution model is as below:

- ``VK_ARM_tensors`` is the direct tensor API path.
- ``VK_ARM_data_graph`` is the graph execution path.
- TOSA, optical flow, and Motion Engine operations are all routed through the
  data-graph path rather than through separate top-level execution APIs.

Vulkan® Extension Map
---------------------

.. list-table::
   :header-rows: 1

   * - Vulkan® extension
     - Exposed by
     - Relationship
     - Detailed page
   * - ``VK_ARM_tensors``
     - ``VK_LAYER_ML_Tensor_Emulation``
     - Foundation for tensor objects, tensor views, and tensor-aware compute
       shaders. Also used by the graph layer.
     - :doc:`tensor`
   * - ``VK_ARM_data_graph``
     - ``VK_LAYER_ML_Graph_Emulation``
     - Core graph pipeline, graph session, and graph dispatch API.
     - :doc:`data_graph`
   * - ``VK_ARM_data_graph_instruction_set_tosa``
     - ``VK_LAYER_ML_Graph_Emulation``
     - Add-on to ``VK_ARM_data_graph`` for the ``TOSA.001000.1`` instruction
       set.
     - :doc:`data_graph`
   * - ``VK_ARM_data_graph_optical_flow``
     - ``VK_LAYER_ML_Graph_Emulation``
     - Add-on to ``VK_ARM_data_graph`` for optical-flow queries and
       optical-flow pipeline nodes.
     - :doc:`data_graph`

SPIR-V™ Map
-----------

.. list-table::
   :header-rows: 1

   * - SPIR-V™ extension or instruction set
     - Consumed by
     - Route through the Emulation Layer
   * - ``SPV_ARM_tensors``
     - Tensor layer
     - Direct tensor compute shaders.
   * - ``SPV_ARM_graph``
     - Graph layer
     - Core graph pipeline representation.
   * - ``TOSA.001000.1``
     - Graph layer
     - Processed through ``VK_ARM_data_graph`` and documented on
       :doc:`data_graph`.
   * - ``Arm.MotionEngine.100``
     - Graph layer
     - Processed through ``VK_ARM_data_graph`` and documented on
       :doc:`data_graph`.

The shared graph processing path also handles reduced-precision tensor element
encodings used by the ML extensions, including ``BFloat16KHR``, ``Float8E5M2EXT``, and
``Float8E4M3EXT``.

Detailed Pages
--------------

Use the following child pages for the detailed API surface. For platform-
specific constraints, see :doc:`limitations`.

.. toctree::
   :maxdepth: 1

   data_graph.rst
   tensor.rst
